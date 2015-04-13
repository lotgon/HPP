#pragma once
#include <string>

struct csv_istream {
	std::istream &is_;
	csv_istream(std::istream &is) : is_(is) {}
	void scan_ws() const {
		while (is_.good()) {
			int c = is_.peek();
			if (c != ' ' && c != '\t') break;
			is_.get();
		}
	}
	void scan(std::string *s = 0) const {
		std::string ws;
		int c = is_.get();
		if (is_.good()) {
			do {
				if (c == ',' || c == '\n') break;
				if (s) {
					ws += c;
					if (c != ' ' && c != '\t') {
						*s += ws;
						ws.clear();
					}
				}
				c = is_.get();
			} while (is_.good());
			if (is_.eof()) is_.clear();
		}
	}
	template <typename T, bool> struct set_value {
		void operator () (std::string in, T &v) const {
			std::istringstream(in) >> v;
		}
	};
	template <typename T> struct set_value<T, true> {
		template <bool SIGNED> void convert(std::string in, T &v) const {
			if (SIGNED) v = ::strtoll(in.c_str(), 0, 0);
			else v = ::strtoull(in.c_str(), 0, 0);
		}
		void operator () (std::string in, T &v) const {
			convert<is_signed_int<T>::val>(in, v);
		}
	};
	template <typename T> const csv_istream & operator >> (T &v) const {
		std::string tmp;
		scan(&tmp);
		set_value<T, is_int<T>::val>()(tmp, v);
		return *this;
	}
	const csv_istream & operator >> (std::string &v) const {
		v.clear();
		scan_ws();
		if (is_.peek() != '"') scan(&v);
		else {
			std::string tmp;
			is_.get();
			std::getline(is_, tmp, '"');
			while (is_.peek() == '"') {
				v += tmp;
				v += is_.get();
				std::getline(is_, tmp, '"');
			}
			v += tmp;
			scan();
		}
		return *this;
	}
	template <typename T>
	const csv_istream & operator >> (T &(*manip)(T &)) const {
		is_ >> manip;
		return *this;
	}
	operator bool() const { return !is_.fail(); }
};

parallel_for_each(D.extent.tile<TILE_SIZE>(), [=](concurrency::tiled_index<TILE_SIZE> t_idx) restrict(amp)
{

	float threshold = P[t_idx.global] + tp;
	float maxDrawdown = 0;
	float open = P[t_idx.global];

	tile_static float my_tile[TILE_SIZE];
	tile_static int calculatedCount;

	int stride = 0;
	bool calculated = false;

	calculatedCount = 0;
	t_idx.barrier.wait();

	for (; calculatedCount < TILE_SIZE; stride++)
	{
		if (t_idx.global[0] + stride*TILE_SIZE < sz)
			my_tile[t_idx.local[0]] = P[t_idx.global + stride*TILE_SIZE];

		t_idx.barrier.wait();

		if (calculated)
			continue;

		for (int i = stride == 0 ? t_idx.local[0] : 0; i < TILE_SIZE; i++)
		{
			if (maxDrawdown < open - my_tile[i])
				maxDrawdown = open - my_tile[i];
			if (my_tile[i] >= threshold)
			{
				duration[t_idx.global] = stride*TILE_SIZE + i - t_idx.local[0];
				D[t_idx.global] = maxDrawdown;
				calculated = true;
				calculatedCount++;
				break;
			}

			if (stride*TILE_SIZE + i >= sz)
			{
				calculated = true;
				calculatedCount++;
				D[t_idx.global] = maxDrawdown;
				duration[t_idx.global] = 0;
				break;
			}
		}
	}
});