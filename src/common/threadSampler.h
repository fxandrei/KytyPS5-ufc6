#ifndef KYTY_COMMON_THREAD_SAMPLER_H_
#define KYTY_COMMON_THREAD_SAMPLER_H_

namespace Common::ThreadSampler {

// Local profiling aid (Windows only). With KYTY_SAMPLE_OUT=<path>, a background thread samples
// the calling thread's call stack about every KYTY_SAMPLE_INTERVAL_US microseconds (default
// 1000) and rewrites <path> every 10 seconds: the loaded modules, then one line per distinct
// stack, "<count> <address>..." with the innermost frame first. While the file named by
// KYTY_SAMPLE_CONTROL holds 0, sampling pauses. KYTY_SAMPLE_THREAD=<name fragment> samples the
// thread with that description instead, such as a guest thread. Symbolize offline with
// llvm-symbolizer.
void StartForCurrentThread();

} // namespace Common::ThreadSampler

#endif /* KYTY_COMMON_THREAD_SAMPLER_H_ */
