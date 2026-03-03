package main

import (
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/parquet-go/parquet-go"
)

const (
	DataPath   = "data/HI_Small" // Change based on DATASET_SIZE
	WindowDays = 3
	StepSize   = 1
	Epsilon    = 1e-9
)

// Raw Parquet Structs (Segment Parquet-Go tags)
type RawAccount struct {
	AccountNumber string `parquet:"Account Number"`
	EntityID      string `parquet:"Entity ID"`
}

type RawTransaction struct {
	Timestamp      int64   `parquet:"timestamp"`
	FromAccount    string  `parquet:"from_account"`
	ToAccount      string  `parquet:"to_account"`
	AmountReceived float64 `parquet:"amount_received"`
	AmountSent     float64 `parquet:"amount_sent_c"`
	IsLaundering   int64   `parquet:"is_laundering"`
}

// Output Parquet Structs
type AggregatedEdge struct {
	WindowDate string  `parquet:"window_date"`
	Source     string  `parquet:"source"`
	Target     string  `parquet:"target"`
	Weight     float64 `parquet:"weight"`
	Volume     float64 `parquet:"volume"`
	Count      int64   `parquet:"count"`
	AmountStd  float64 `parquet:"amount_std"`
}

type AggregatedNode struct {
	WindowDate string  `parquet:"window_date"`
	EntityID   string  `parquet:"entity_id"`
	VolSent    float64 `parquet:"vol_sent"`
	VolRecv    float64 `parquet:"vol_recv"`
	TxCount    int64   `parquet:"tx_count"`
}

// Internal Structs
type ResolvedTransaction struct {
	Timestamp      time.Time
	SourceEntity   string
	TargetEntity   string
	AmountSentC    float64
	AmountReceived float64
	IsLaundering   int64
}

type WindowResult struct {
	Edges []AggregatedEdge
	Nodes []AggregatedNode
}

func main() {
	log.Println("Starting Data Engine...")

	// 1. Load Accounts & Build Mapping
	accountMap := make(map[string][]string)

	fAcct, err := os.Open(filepath.Join(DataPath, "3_filtered_accounts.parquet"))
	if err != nil { log.Fatal(err) }
	
	acctReader := parquet.NewGenericReader[RawAccount](fAcct)
	acctBuffer := make([]RawAccount, 2000)
	for {
		n, err := acctReader.Read(acctBuffer)
		for i := 0; i < n; i++ {
			acc := acctBuffer[i]
			accountMap[acc.AccountNumber] = append(accountMap[acc.AccountNumber], acc.EntityID)
		}
		if err == io.EOF { break }
		if err != nil { log.Fatal(err) }
	}
	acctReader.Close()
	fAcct.Close()

	// 2. Load Transactions
	var resolvedTxns []ResolvedTransaction
	var minDate, maxDate time.Time
	firstTx := true

	loadTxns := func(filename string) {
		fTx, err := os.Open(filepath.Join(DataPath, filename))
		if err != nil { log.Fatal(err) }
		defer fTx.Close()

		txReader := parquet.NewGenericReader[RawTransaction](fTx)
		defer txReader.Close()

		txBuffer := make([]RawTransaction, 5000)
		for {
			n, err := txReader.Read(txBuffer)
			for i := 0; i < n; i++ {
				tx := txBuffer[i]
				srcEntities, srcOk := accountMap[tx.FromAccount]
				tgtEntities, tgtOk := accountMap[tx.ToAccount]

				if !srcOk || !tgtOk { continue }

				// Decode PyArrow INT64 timestamp robustly in UTC
				var ts time.Time
				if tx.Timestamp > 1e16 {
					ts = time.Unix(0, tx.Timestamp).UTC() // Nanoseconds
				} else if tx.Timestamp > 1e13 {
					ts = time.UnixMicro(tx.Timestamp).UTC() // Microseconds
				} else if tx.Timestamp > 1e10 {
					ts = time.UnixMilli(tx.Timestamp).UTC() // Milliseconds
				} else {
					ts = time.Unix(tx.Timestamp, 0).UTC() // Seconds
				}

				if firstTx {
					minDate, maxDate = ts, ts
					firstTx = false
				} else {
					if ts.Before(minDate) { minDate = ts }
					if ts.After(maxDate) { maxDate = ts }
				}

				divisor := float64(len(srcEntities) * len(tgtEntities))
				adjSent := tx.AmountSent / divisor
				adjRecv := tx.AmountReceived / divisor

				for _, src := range srcEntities {
					for _, tgt := range tgtEntities {
						resolvedTxns = append(resolvedTxns, ResolvedTransaction{
							Timestamp:      ts,
							SourceEntity:   src,
							TargetEntity:   tgt,
							AmountSentC:    adjSent,
							AmountReceived: adjRecv,
							IsLaundering:   tx.IsLaundering,
						})
					}
				}
			}
			if err == io.EOF { break }
			if err != nil { log.Fatal(err) }
		}
	}

	loadTxns("1_filtered_normal_transactions.parquet")
	loadTxns("2_filtered_laundering_transactions.parquet")

	// 3. Sliding Window Generation (Concurrent)
	minDate = minDate.UTC().Truncate(24 * time.Hour)
	maxDate = maxDate.UTC().Truncate(24 * time.Hour)

	log.Printf("Loaded %d resolved transactions. Range: %v to %v", len(resolvedTxns), minDate.Format("2006-01-02"), maxDate.Format("2006-01-02"))

	// 3. Sliding Window Generation (Concurrent)
	currentDate := minDate.Add(24 * time.Hour)
	endDate := maxDate

	resultChan := make(chan WindowResult, 100)
	var wg sync.WaitGroup
	sem := make(chan struct{}, 8)

	for currentDate.Before(endDate) || currentDate.Equal(endDate) {
		wg.Add(1)
		sem <- struct{}{}

		go func(cd time.Time) {
			defer wg.Done()
			defer func() { <-sem }()

			windowStart := cd.AddDate(0, 0, -WindowDays)
			var windowTxns []ResolvedTransaction
			for _, tx := range resolvedTxns {
				if tx.Timestamp.After(windowStart) && !tx.Timestamp.After(cd) {
					windowTxns = append(windowTxns, tx)
				}
			}

			if len(windowTxns) == 0 { return }

			dateStr := cd.Format("2006-01-02")
			edges, nodes := processWindow(dateStr, windowTxns)
			
			resultChan <- WindowResult{Edges: edges, Nodes: nodes}

		}(currentDate)

		currentDate = currentDate.AddDate(0, 0, StepSize)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 4. Collect and Write Parquet
	fEdges, err := os.Create(filepath.Join(DataPath, "aggregated_edges.parquet"))
	if err != nil { log.Fatal(err) }
	edgeWriter := parquet.NewGenericWriter[AggregatedEdge](fEdges)
	
	fNodes, err := os.Create(filepath.Join(DataPath, "aggregated_nodes.parquet"))
	if err != nil { log.Fatal(err) }
	nodeWriter := parquet.NewGenericWriter[AggregatedNode](fNodes)

	var edgeCount, nodeCount int
	for res := range resultChan {
		if len(res.Edges) > 0 {
			_, err = edgeWriter.Write(res.Edges)
			if err != nil { log.Fatal(err) }
			edgeCount += len(res.Edges)
		}
		if len(res.Nodes) > 0 {
			_, err = nodeWriter.Write(res.Nodes)
			if err != nil { log.Fatal(err) }
			nodeCount += len(res.Nodes)
		}
	}

	edgeWriter.Close()
	fEdges.Close()
	nodeWriter.Close()
	fNodes.Close()

	log.Printf("Pipeline complete. Wrote %d edges and %d node records.", edgeCount, nodeCount)
}

func processWindow(dateStr string, txns []ResolvedTransaction) ([]AggregatedEdge, []AggregatedNode) {
	type EdgeKey struct{ Src, Tgt string }
	edgeData := make(map[EdgeKey][]float64)
	
	type NodeData struct {
		Sent  float64
		Recv  float64
		Count int64
	}
	nodeMap := make(map[string]*NodeData)

	for _, tx := range txns {
		k := EdgeKey{tx.SourceEntity, tx.TargetEntity}
		edgeData[k] = append(edgeData[k], tx.AmountSentC)

		if _, ok := nodeMap[tx.SourceEntity]; !ok { nodeMap[tx.SourceEntity] = &NodeData{} }
		if _, ok := nodeMap[tx.TargetEntity]; !ok { nodeMap[tx.TargetEntity] = &NodeData{} }

		nodeMap[tx.SourceEntity].Sent += tx.AmountSentC
		nodeMap[tx.SourceEntity].Count++
		nodeMap[tx.TargetEntity].Recv += tx.AmountSentC
		nodeMap[tx.TargetEntity].Count++
	}

	var edges []AggregatedEdge
	for k, amounts := range edgeData {
		count := float64(len(amounts))
		sum := 0.0
		for _, a := range amounts { sum += a }
		mean := sum / count

		std := 0.0
		if count > 1 {
			variance := 0.0
			for _, a := range amounts {
				variance += (a - mean) * (a - mean)
			}
			std = math.Sqrt(variance / (count - 1))
		}

		cv := std / (mean + Epsilon)
		weight := sum * math.Log2(1+count) * (1 + 1.0/(1.0+cv))

		edges = append(edges, AggregatedEdge{
			WindowDate: dateStr,
			Source:     k.Src,
			Target:     k.Tgt,
			Weight:     weight,
			Volume:     sum,
			Count:      int64(count),
			AmountStd:  std,
		})
	}

	var nodes []AggregatedNode
	for id, stats := range nodeMap {
		nodes = append(nodes, AggregatedNode{
			WindowDate: dateStr,
			EntityID:   id,
			VolSent:    stats.Sent,
			VolRecv:    stats.Recv,
			TxCount:    stats.Count,
		})
	}

	return edges, nodes
}