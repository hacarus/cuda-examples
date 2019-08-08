#include <chrono>

class perf_counter {
 public:
	using clock = std::chrono::steady_clock;
	// void clear() { start(); tse = tsb; }
	void start() { tsb = clock::now(); tse = tsb; }
	void stop()  { tse = clock::now(); }
	double nsecs() const{
		using namespace std::chrono;
		return duration_cast<nanoseconds>(tse - tsb).count();
	}
	double usecs() const { return double(nsecs()) / 1000.0; }
	double msecs() const { return double(nsecs()) / 1000000.0; }
	double  secs() const { return double(nsecs()) / 1000000000.0; }
	// Return msec
	friend std::ostream& operator<<(std::ostream& o, perf_counter const& timer) {
		return o << timer.msecs();
	}
 private:
	clock::time_point tsb;
	clock::time_point tse;
};


