// deep_recursion.go
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
	"time"
)

// ---- CSV helper ----

type BatchCSV struct {
	writer *csv.Writer
}

func (c *BatchCSV) Header() {
	if c.writer != nil {
		_ = c.writer.Write([]string{"mode", "root", "depth", "fanout", "ms", "goroutines", "checksum"})
	}
}
func (c *BatchCSV) Row(mode string, root, depth, fanout int, dur time.Duration, checksum uint64) {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			mode,
			fmt.Sprint(root),
			fmt.Sprint(depth),
			fmt.Sprint(fanout),
			fmt.Sprintf("%.3f", float64(dur.Nanoseconds())/1e6),
			fmt.Sprint(runtime.NumGoroutine()),
			fmt.Sprint(checksum),
		})
	}
}
func (c *BatchCSV) Flush() { if c.writer != nil { c.writer.Flush() } }

// ---- Workloads (recursion) ----

// tiny CPU spin so the compiler can’t optimize the recursion away
func spinWork(spin int) {
	if spin <= 0 {
		return
	}
	s := 0
	for i := 0; i < spin; i++ {
		s += i
	}
	_ = s
}

func linearRec(depth, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	return 1 + linearRec(depth-1, spin)
}

func binaryRec(depth, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	// 1 + left + right
	return 1 + binaryRec(depth-1, spin) + binaryRec(depth-2, spin)
}

func karyRec(depth, fanout, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	var sum uint64 = 1
	for i := 0; i < fanout; i++ {
		sum += karyRec(depth-1, fanout, spin)
	}
	return sum
}

// ---- Runner ----

type result struct {
	rootIdx   int
	checksum  uint64
	elapsed   time.Duration
}

func runRoots(mode string, roots, depth, fanout, spin int) (results []result) {
	results = make([]result, 0, roots)
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}

	for r := 0; r < roots; r++ {
		wg.Add(1)
		go func(root int) {
			defer wg.Done()
			t0 := time.Now()
			var chk uint64
			switch mode {
			case "linear":
				chk = linearRec(depth, spin)
			case "binary":
				chk = binaryRec(depth, spin)
			case "kary":
				chk = karyRec(depth, fanout, spin)
			default:
				chk = linearRec(depth, spin)
			}
			el := time.Since(t0)
			mu.Lock()
			results = append(results, result{rootIdx: root, checksum: chk, elapsed: el})
			mu.Unlock()
		}(r)
	}
	wg.Wait()
	return
}

// ---- Durations helpers ----

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
	// insertion sort (roots count is usually small)
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

// ---- main ----

func main() {
	// Workload controls
	mode := flag.String("mode", "linear", "Recursion mode: linear | binary | kary")
	depth := flag.Int("depth", 50000, "Recursion depth (CAUTION: large values can crash)")
	fanout := flag.Int("fanout", 3, "Fanout for k-ary recursion (ignored unless mode=kary)")
	roots := flag.Int("roots", 8, "Number of independent recursion roots (run in parallel)")
	spin := flag.Int("spin", 0, "Artificial CPU work per call (inner loop iterations)")

	// Profiling
	cpuprofile := flag.String("cpuprofile", "", "Write CPU profile to file")
	memprofile := flag.String("memprofile", "", "Write memory profile to file at end")
	traceFile := flag.String("trace", "", "Write runtime trace to file")
	httpAddr := flag.String("http", "", "Start pprof HTTP server at address (e.g. :6060)")

	// Output
	csvPath := flag.String("csv", "", "Write per-root timings CSV")
	printEvery := flag.Int("print_every", 2, "Print every k roots (0=never)")

	flag.Parse()

	// Sanity defaults to avoid explosions
	switch *mode {
	case "binary":
		if flag.Lookup("depth").Value.String() == "50000" { // default unchanged
			*depth = 28 // ~2^28 calls (still heavy!)
		}
	case "kary":
		if flag.Lookup("depth").Value.String() == "50000" {
			*depth = 12
		}
		if *fanout < 2 {
			*fanout = 2
		}
	}

	// Live pprof
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

	// Trace
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

	// CSV
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

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	start := time.Now()

	// Run
	log.Printf("Mode=%s depth=%d fanout=%d roots=%d spin=%d", *mode, *depth, *fanout, *roots, *spin)
	results := runRoots(*mode, *roots, *depth, *fanout, *spin)
	totalWall := time.Since(start)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Summaries
	durs := make([]time.Duration, 0, len(results))
	var aggChecksum uint64
	for i, r := range results {
		if csvOut.writer != nil {
			csvOut.Row(*mode, r.rootIdx, *depth, *fanout, r.elapsed, r.checksum)
		}
		if *printEvery > 0 && (i%*printEvery == 0 || i == len(results)-1) {
			log.Printf("root=%d dur=%s checksum=%d", r.rootIdx, r.elapsed, r.checksum)
		}
		aggChecksum ^= (r.checksum + uint64(i)*1315423911)
		durs = append(durs, r.elapsed)
	}

	sum, min, max := summarizeDurations(durs)
	avg := time.Duration(int64(sum) / int64(len(durs)))
	pcts := percentiles(durs, []float64{0.50, 0.90, 0.99})

	fmt.Println("==== SUMMARY ====")
	fmt.Printf("Mode=%s Depth=%d Fanout=%d Roots=%d Spin=%d\n", *mode, *depth, *fanout, *roots, *spin)
	fmt.Printf("Total wall: %s  per-root avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		totalWall, avg, min, pcts[0], pcts[1], pcts[2], max)
	fmt.Printf("Aggregate checksum: %d\n", aggChecksum)

	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Goroutines (end): %d\n", runtime.NumGoroutine())
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)

	// Optional heap profile at end
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
}
