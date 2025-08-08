// series_bench.go
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"sync/atomic"
	"time"
)

// ----------------------- Flags & main -----------------------

type BatchCSV struct {
	writer *csv.Writer
}

func (c *BatchCSV) Header() {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			"phase", "batch", "batch_ms", "goroutines", "checksum",
		})
	}
}
func (c *BatchCSV) Row(phase string, batch int, dur time.Duration, checksum uint64) {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			phase,
			fmt.Sprint(batch),
			fmt.Sprintf("%.3f", float64(dur.Nanoseconds())/1e6),
			fmt.Sprint(runtime.NumGoroutine()),
			fmt.Sprint(checksum),
		})
	}
}
func (c *BatchCSV) Flush() {
	if c.writer != nil {
		c.writer.Flush()
	}
}

func main() {
	// Work sizes
	oddsN := flag.Int("oddsN", 1_000_000, "Upper limit for odd enumeration (inclusive)")
	fibN := flag.Int("fibN", 1_000_000, "Number of Fibonacci terms to generate")
	fibMod := flag.Uint64("fibMod", 1_000_000_007, "Modulo for Fibonacci to avoid overflow")
	fibStore := flag.Bool("fibStore", false, "Store Fibonacci sequence in memory (stress heap)")
	fibRepeat := flag.Int("fibRepeat", 1, "Repeat Fibonacci generation this many times")

	// Parallelism & batching
	workers := flag.Int("workers", 0, "Number of worker goroutines for Odds (0 => GOMAXPROCS)")
	chunks := flag.Int("chunks", 0, "Number of work chunks (0 => auto)")
	spin := flag.Int("spin", 0, "Artificial inner loop per iteration to simulate extra work")

	// Profiling & telemetry
	cpuprofile := flag.String("cpuprofile", "", "Write CPU profile to file")
	memprofile := flag.String("memprofile", "", "Write memory profile to file at end")
	traceFile := flag.String("trace", "", "Write runtime trace to file")
	httpAddr := flag.String("http", "", "Start pprof HTTP server at address (e.g. :6060)")
	csvPath := flag.String("csv", "", "Write per-batch timings CSV")
	printEvery := flag.Int("print_every", 5, "Print phase progress every k batches (0=never)")

	flag.Parse()

	// Live pprof server
	if *httpAddr != "" {
		go func() {
			log.Printf("pprof HTTP server on %s (see /debug/pprof/)", *httpAddr)
			if err := http.ListenAndServe(*httpAddr, nil); err != nil {
				log.Printf("pprof server error: %v", err)
			}
		}()
	}

	// CPU profile
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpu profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Runtime trace
	if *traceFile != "" {
		tf, err := os.Create(*traceFile)
		if err != nil {
			log.Fatalf("create trace: %v", err)
		}
		defer tf.Close()
		if err := trace.Start(tf); err != nil {
			log.Fatalf("start trace: %v", err)
		}
		defer trace.Stop()
	}

	// CSV init (avoid shadowing the csv package)
	var csvOut BatchCSV
	if *csvPath != "" {
		f, err := os.Create(*csvPath)
		if err != nil {
			log.Fatalf("create csv: %v", err)
		}
		defer f.Close()
		bw := bufio.NewWriter(f)
		defer bw.Flush()
		csvOut.writer = csv.NewWriter(bw)
		defer csvOut.writer.Flush()
		csvOut.Header()
	}

	// Worker defaults
	if *workers <= 0 {
		*workers = runtime.GOMAXPROCS(0)
	}
	if *chunks <= 0 {
		*chunks = *workers * 4
	}

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	globalStart := time.Now()

	// ----------------------- Phase 1: Odds -----------------------
	log.Printf("Phase: Odds  N=%d  workers=%d chunks=%d spin=%d", *oddsN, *workers, *chunks, *spin)
	oddsStart := time.Now()
	oddsSum, oddsCnt, oddsBatches, oddsTimes := runOdds(*oddsN, *workers, *chunks, *spin, &csvOut, *printEvery)
	oddsWall := time.Since(oddsStart)
	log.Printf("Odds done: sum=%d count=%d wall=%s", oddsSum, oddsCnt, oddsWall)

	// ----------------------- Phase 2: Fibonacci -----------------------
	log.Printf("Phase: Fibonacci  terms=%d mod=%d store=%v repeat=%d", *fibN, *fibMod, *fibStore, *fibRepeat)
	fibStart := time.Now()
	fibChecksum, fibBatches, fibTimes := runFibonacci(*fibN, *fibMod, *fibStore, *fibRepeat, &csvOut, *printEvery)
	fibWall := time.Since(fibStart)
	log.Printf("Fibonacci done: checksum=%d wall=%s", fibChecksum, fibWall)

	totalWall := time.Since(globalStart)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Optional heap profile
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write heap profile: %v", err)
		}
	}

	// Summaries
	fmt.Println("==== SUMMARY ====")
	fmt.Printf("GOMAXPROCS: %d  Goroutines(end): %d\n", runtime.GOMAXPROCS(0), runtime.NumGoroutine())
	fmt.Printf("Total wall: %s  (Odds: %s, Fibonacci: %s)\n", totalWall, oddsWall, fibWall)

	oddsSumDur, oddsMin, oddsMax := summarizeDurations(oddsTimes)
	oddsAvg := time.Duration(int64(oddsSumDur) / int64(len(oddsTimes)))
	oddsP := percentiles(oddsTimes, []float64{0.5, 0.9, 0.99})

	fibSumDur, fibMin, fibMax := summarizeDurations(fibTimes)
	fibAvg := time.Duration(int64(fibSumDur) / int64(len(fibTimes)))
	fibP := percentiles(fibTimes, []float64{0.5, 0.9, 0.99})

	fmt.Println("-- Odds batches --")
	fmt.Printf("batches: %d  avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		oddsBatches, oddsAvg, oddsMin, oddsP[0], oddsP[1], oddsP[2], oddsMax)

	fmt.Println("-- Fibonacci batches --")
	fmt.Printf("batches: %d  avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		fibBatches, fibAvg, fibMin, fibP[0], fibP[1], fibP[2], fibMax)

	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)

	_ = oddsBatches
	_ = fibBatches
}

// ----------------------- Odds (parallel) -----------------------

func runOdds(N int, workers, chunks, spin int, csvOut *BatchCSV, printEvery int) (sum uint64, count int64, batches int, batchDurations []time.Duration) {
	if N < 1 {
		return 0, 0, 0, nil
	}
	// Batch the outer range so CSV/percentiles are meaningful.
	batches = chunks
	if batches < 1 {
		batches = 1
	}
	chunkSize := (N + batches - 1) / batches
	type job struct{ lo, hi int }

	jobs := make(chan job, batches)
	type res struct {
		sum   uint64
		count int64
		dur   time.Duration
	}
	results := make(chan res, batches)

	var globalSum uint64
	var globalCount int64
	var wg sync.WaitGroup

	// Workers
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				var localSum uint64
				var localCnt int64
				// Enumerate odds in [j.lo .. j.hi]
				start := j.lo
				if start%2 == 0 {
					start++
				}
				for i := start; i <= j.hi; i += 2 {
					// cheap spin to simulate extra CPU work if requested
					if spin > 0 {
						var s int
						for k := 0; k < spin; k++ {
							s += k
						}
						_ = s
					}
					localSum += uint64(i)
					localCnt++
				}
				results <- res{
					sum:   localSum,
					count: localCnt,
					dur:   time.Since(t0),
				}
			}
		}()
	}

	// Enqueue batches
	for lo := 1; lo <= N; lo += chunkSize {
		hi := lo + chunkSize - 1
		if hi > N {
			hi = N
		}
		jobs <- job{lo: lo, hi: hi}
	}
	close(jobs)

	// Collect
	batchDurations = make([]time.Duration, 0, batches)
	done := make(chan struct{})
	go func() {
		defer close(done)
		idx := 0
		for r := range results {
			atomic.AddUint64(&globalSum, r.sum)
			atomic.AddInt64(&globalCount, r.count)
			batchDurations = append(batchDurations, r.dur)
			if csvOut != nil {
				csvOut.Row("odds", idx, r.dur, r.sum)
			}
			if printEvery > 0 && idx%printEvery == 0 {
				log.Printf("Odds batch %d dur=%s partial_sum=%d", idx, r.dur, r.sum)
			}
			idx++
		}
	}()

	wg.Wait()
	close(results)
	<-done

	return globalSum, globalCount, len(batchDurations), batchDurations
}

// ----------------------- Fibonacci (sequential batches) -----------------------

func runFibonacci(N int, mod uint64, store bool, repeat int, csvOut *BatchCSV, printEvery int) (checksum uint64, batches int, batchDurations []time.Duration) {
	if N <= 0 {
		return 0, 0, nil
	}
	// We'll split the sequence into fixed-size batches so we can time them.
	const targetBatches = 32
	batches = targetBatches
	batchSize := (N + batches - 1) / batches
	if batchSize < 1 {
		batchSize = 1
	}
	// Recompute batches with new size to cover exactly N
	batches = (N + batchSize - 1) / batchSize

	batchDurations = make([]time.Duration, 0, batches*repeat)

	for r := 0; r < repeat; r++ {
		var a, b uint64 = 0, 1
		var storeBuf []uint64
		if store {
			storeBuf = make([]uint64, 0, N)
			storeBuf = append(storeBuf, a, b)
		}
		idx := 0
		written := 2

		for batch := 0; batch < batches; batch++ {
			t0 := time.Now()
			limit := (batch + 1) * batchSize
			if limit > N {
				limit = N
			}
			for idx < limit {
				a, b = b%mod, (a+b)%mod
				if store {
					if written < N {
						storeBuf = append(storeBuf, a)
						written++
					}
				}
				idx++
			}
			dur := time.Since(t0)
			if csvOut != nil {
				csvOut.Row("fibonacci", batch+(r*batches), dur, a^b)
			}
			if printEvery > 0 && (batch%printEvery == 0 || batch == batches-1) {
				log.Printf("Fib rep=%d batch=%d/%d dur=%s last_pair=(%d,%d)", r, batch+1, batches, dur, a, b)
			}
			batchDurations = append(batchDurations, dur)
		}
		// Accumulate checksum so the loop isn’t optimized out
		checksum ^= a + 31*b + uint64(len(batchDurations))
		// keep storeBuf in scope (and printed size) so it isn’t DCE’d
		if store {
			log.Printf("Fib rep=%d stored_terms=%d", r, len(storeBuf))
		}
	}

	return checksum, len(batchDurations), batchDurations
}

// ----------------------- Timing helpers -----------------------

func summarizeDurations(dd []time.Duration) (sum, min, max time.Duration) {
	if len(dd) == 0 {
		return 0, 0, 0
	}
	min, max = dd[0], dd[0]
	for _, d := range dd {
		sum += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return
}

func percentiles(dd []time.Duration, ps []float64) []time.Duration {
	if len(dd) == 0 {
		return make([]time.Duration, len(ps))
	}
	cp := make([]time.Duration, len(dd))
	copy(cp, dd)
	// insertion sort
	for i := 1; i < len(cp); i++ {
		k := cp[i]
		j := i - 1
		for j >= 0 && cp[j] > k {
			cp[j+1] = cp[j]
			j--
		}
		cp[j+1] = k
	}
	res := make([]time.Duration, 0, len(ps))
	for _, p := range ps {
		switch {
		case p <= 0:
			res = append(res, cp[0])
		case p >= 1:
			res = append(res, cp[len(cp)-1])
		default:
			idx := int(math.Round(p*float64(len(cp)-1)))
			res = append(res, cp[idx])
		}
	}
	return res
}
