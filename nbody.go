// nbody.go
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
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

const G = 6.67430e-11

type Vec3 struct{ X, Y, Z float64 }

func (a Vec3) Add(b Vec3) Vec3      { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Sub(b Vec3) Vec3      { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a Vec3) Scale(s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func (a Vec3) Dot(b Vec3) float64   { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a Vec3) Norm() float64        { return math.Sqrt(a.Dot(a)) }

type Body struct {
	Pos  Vec3
	Vel  Vec3
	Mass float64
	F    Vec3 // force accumulator
}

type StepStat struct {
	Step          int
	StepWallNanos int64
	KE            float64
	PE            float64
	TotalE        float64
	PMag          float64
	Goroutines    int
}

func main() {
	// --- Flags ---
	n := flag.Int("n", 2000, "number of bodies")
	steps := flag.Int("steps", 200, "simulation steps")
	dt := flag.Float64("dt", 1e-3, "time step (seconds)")
	softening := flag.Float64("softening", 1e-3, "softening factor (epsilon)")
	seed := flag.Int64("seed", 42, "random seed (negative to use time-based)")
	workers := flag.Int("workers", 0, "number of worker goroutines (0 => GOMAXPROCS)")
	chunks := flag.Int("chunks", 0, "number of work chunks (0 => auto)")
	velScale := flag.Float64("velscale", 1e-3, "initial velocity scale")
	posScale := flag.Float64("posscale", 1.0, "initial position scale")

	cpuprofile := flag.String("cpuprofile", "", "write CPU profile to file")
	memprofile := flag.String("memprofile", "", "write memory profile to file (at end)")
	traceFile := flag.String("trace", "", "write runtime trace to file")
	httpAddr := flag.String("http", "", "start pprof HTTP server at address (e.g. :6060)")
	csvFile := flag.String("csv", "", "write per-step timings to CSV file")
	verify := flag.Bool("verify", true, "compute energy & momentum each step (PE is O(n^2))")
	printEvery := flag.Int("print_every", 20, "print stats every k steps (0=never)")
	flag.Parse()

	// Optional pprof HTTP server
	if *httpAddr != "" {
		go func() {
			log.Printf("pprof HTTP server listening on %s (visit /debug/pprof/)", *httpAddr)
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

	// Random seed
	if *seed < 0 {
		*seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(*seed))
	log.Printf("Seed: %d", *seed)

	// Decide workers/chunks
	if *workers <= 0 {
		*workers = runtime.GOMAXPROCS(0)
	}
	if *chunks <= 0 {
		*chunks = *workers * 4
	}

	// Init system
	bodies := make([]Body, *n)
	initBodies(bodies, rng, *posScale, *velScale)

	// CSV
	var csvWriter *csv.Writer
	if *csvFile != "" {
		f, err := os.Create(*csvFile)
		if err != nil {
			log.Fatalf("create csv: %v", err)
		}
		defer f.Close()
		bw := bufio.NewWriter(f)
		defer bw.Flush()
		csvWriter = csv.NewWriter(bw)
		_ = csvWriter.Write([]string{"step", "step_ms", "ke", "pe", "total_e", "momentum", "goroutines"})
		defer csvWriter.Flush()
	}

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	start := time.Now()

	// Worker pool
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	forcePool := newForcePool(*workers, *chunks, *softening)

	// Simulation loop
	stepTimes := make([]time.Duration, *steps)
	var totalPairs uint64
	for s := 0; s < *steps; s++ {
		t0 := time.Now()

		// Compute forces (O(n^2), chunked)
		pairs := forcePool.ComputeForces(ctx, bodies)
		atomic.AddUint64(&totalPairs, uint64(pairs))

		// Integrate (semi-implicit Euler)
		integrate(bodies, *dt)

		stepTimes[s] = time.Since(t0)

		// Optional verification & metrics
		var ke, pe, pmag float64
		if *verify {
			ke = kineticEnergy(bodies)
			pe = potentialEnergy(bodies, *softening)
			pmag = momentumMag(bodies)
		}
		stats := StepStat{
			Step:          s,
			StepWallNanos: stepTimes[s].Nanoseconds(),
			KE:            ke,
			PE:            pe,
			TotalE:        ke + pe,
			PMag:          pmag,
			Goroutines:    runtime.NumGoroutine(),
		}

		if csvWriter != nil {
			_ = csvWriter.Write([]string{
				fmt.Sprint(stats.Step),
				fmt.Sprintf("%.3f", float64(stats.StepWallNanos)/1e6),
				fmt.Sprintf("%.6e", stats.KE),
				fmt.Sprintf("%.6e", stats.PE),
				fmt.Sprintf("%.6e", stats.TotalE),
				fmt.Sprintf("%.6e", stats.PMag),
				fmt.Sprint(stats.Goroutines),
			})
		}

		if *printEvery > 0 && (s%*printEvery == 0 || s == *steps-1) {
			log.Printf("step=%d dt=%.2e step_ms=%.3f KE=%.3e PE=%.3e E=%.3e |P|=%.3e goroutines=%d",
				s, *dt, float64(stepTimes[s].Microseconds())/1000.0, ke, pe, ke+pe, pmag, stats.Goroutines)
		}
	}
	total := time.Since(start)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Profiling dumps
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
	sum, min, max := summarizeDurations(stepTimes)
	avg := time.Duration(int64(sum) / int64(len(stepTimes)))
	pcts := percentiles(stepTimes, []float64{0.50, 0.90, 0.99})
	p50, p90, p99 := pcts[0], pcts[1], pcts[2]

	fmt.Println("==== SUMMARY ====")
	fmt.Printf("Bodies: %d, Steps: %d, workers: %d, chunks: %d\n", *n, *steps, *workers, *chunks)
	fmt.Printf("Wall: %s, per-step avg=%s min=%s p50=%s p90=%s p99=%s max=%s\n", total, avg, min, p50, p90, p99, max)
	fmt.Printf("Pairs computed: %d (expected ~ n*(n-1)/2 per step)\n", atomic.LoadUint64(&totalPairs))

	// GC / Mem
	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Goroutines (end): %d\n", runtime.NumGoroutine())
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)
}

// ---------------- Initialization ----------------

func initBodies(b []Body, rng *rand.Rand, posScale, velScale float64) {
	n := len(b)
	for i := 0; i < n; i++ {
		b[i].Pos = Vec3{
			X: (rng.Float64()*2 - 1) * posScale,
			Y: (rng.Float64()*2 - 1) * posScale,
			Z: (rng.Float64()*2 - 1) * posScale,
		}
		b[i].Vel = Vec3{
			X: (rng.Float64()*2 - 1) * velScale,
			Y: (rng.Float64()*2 - 1) * velScale,
			Z: (rng.Float64()*2 - 1) * velScale,
		}
		b[i].Mass = 1.0 + 0.1*(rng.Float64()*2-1)
	}
}

// ---------------- Physics ----------------

func integrate(b []Body, dt float64) {
	for i := range b {
		a := b[i].F.Scale(1.0 / b[i].Mass)
		b[i].Vel = b[i].Vel.Add(a.Scale(dt))
		b[i].Pos = b[i].Pos.Add(b[i].Vel.Scale(dt))
		b[i].F = Vec3{} // reset for next step
	}
}

func kineticEnergy(b []Body) float64 {
	var ke float64
	for i := range b {
		v2 := b[i].Vel.Dot(b[i].Vel)
		ke += 0.5 * b[i].Mass * v2
	}
	return ke
}

func potentialEnergy(b []Body, eps float64) float64 {
	var pe float64
	n := len(b)
	eps2 := eps * eps
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			r := b[j].Pos.Sub(b[i].Pos)
			dist := math.Sqrt(r.Dot(r) + eps2)
			pe += -G * b[i].Mass * b[j].Mass / dist
		}
	}
	return pe
}

func momentumMag(b []Body) float64 {
	var p Vec3
	for i := range b {
		p = p.Add(b[i].Vel.Scale(b[i].Mass))
	}
	return p.Norm()
}

// ---------------- Parallel Force Computation ----------------

type forcePool struct {
	workers int
	chunks  int
	eps     float64
}

func newForcePool(workers, chunks int, eps float64) *forcePool {
	return &forcePool{workers: workers, chunks: chunks, eps: eps}
}

func (fp *forcePool) ComputeForces(ctx context.Context, b []Body) int {
	n := len(b)
	if n == 0 {
		return 0
	}
	chunkSize := (n + fp.chunks - 1) / fp.chunks
	type job struct{ i0, i1 int }
	jobs := make(chan job, fp.chunks)
	var pairs int64

	wg := sync.WaitGroup{}
	wg.Add(fp.workers)
	for w := 0; w < fp.workers; w++ {
		go func() {
			defer wg.Done()
			eps2 := fp.eps * fp.eps
			for j := range jobs {
				for i := j.i0; i < j.i1; i++ {
					fi := Vec3{}
					pi := b[i].Pos
					mi := b[i].Mass
					for k := 0; k < n; k++ {
						if k == i {
							continue
						}
						r := b[k].Pos.Sub(pi)
						d2 := r.Dot(r) + eps2
						invD := 1.0 / math.Sqrt(d2)
						invD3 := invD * invD * invD
						f := r.Scale(G * mi * b[k].Mass * invD3)
						fi = fi.Add(f)
					}
					// i is unique to this goroutine within its chunk → no race on b[i].F
					b[i].F = b[i].F.Add(fi)
					atomic.AddInt64(&pairs, int64(n-1))
				}
			}
		}()
	}

	for i := 0; i < n; i += chunkSize {
		j := i + chunkSize
		if j > n {
			j = n
		}
		select {
		case jobs <- job{i0: i, i1: j}:
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return int(atomic.LoadInt64(&pairs))
		}
	}
	close(jobs)
	wg.Wait()
	return int(atomic.LoadInt64(&pairs))
}

// ---------------- Timing helpers ----------------

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
	cp := make([]time.Duration, len(dd))
	copy(cp, dd)
	// insertion sort (steps is typically modest)
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
