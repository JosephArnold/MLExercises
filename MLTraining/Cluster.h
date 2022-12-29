#pragma once
#include"Data.h"

template <class T>
class Cluster
{
public:
	data<T> center;
	uint64_t size = 0;

	int32_t getClusterLablel() {
		return this.center.getClusterInfo();
	}

};

