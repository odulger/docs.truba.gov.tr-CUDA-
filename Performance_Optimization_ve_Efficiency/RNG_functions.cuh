//Ref: Dülger, Ö., Oğuztüzün, H. & Demirekler, M. Memory Coalescing Implementation of Metropolis Resampling on Graphics Processing Unit. J Sign Process Syst 90, 433–447 (2018)
#ifndef RNG_FUNCTIONS_CUH
#define RNG_FUNCTIONS_CUH

// Class for random number generators
class curandInitializer
{
	public:
		// Constructor
		curandInitializer(const unsigned int _N)
		{
			N = _N;
			isCopy = false;

			cudaMalloc((void**)&d,sizeof(unsigned int)*N);
			cudaMalloc((void**)&v0,sizeof(unsigned int)*N);
			cudaMalloc((void**)&v1,sizeof(unsigned int)*N);
			cudaMalloc((void**)&v2,sizeof(unsigned int)*N);
			cudaMalloc((void**)&v3,sizeof(unsigned int)*N);
			cudaMalloc((void**)&v4,sizeof(unsigned int)*N);
			cudaMalloc((void**)&boxmuller_flag,sizeof(int)*N);
			cudaMalloc((void**)&boxmuller_flag_double,sizeof(int)*N);
			cudaMalloc((void**)&boxmuller_extra,sizeof(float)*N);
			cudaMalloc((void**)&boxmuller_extra_double,sizeof(double)*N);
		}

		// Second constructor
		curandInitializer(const curandInitializer& o) : d(o.d), v0(o.v0), v1(o.v1), v2(o.v2), v3(o.v3), v4(o.v4), boxmuller_flag(
        	o.boxmuller_flag), boxmuller_flag_double(o.boxmuller_flag_double), boxmuller_extra(
        	o.boxmuller_extra), boxmuller_extra_double(o.boxmuller_extra_double),isCopy(true){ }

		// Loading the states of the random number generator
		__device__ void load(curandState_t &state,const unsigned  int i) const
		{
			state.d = d[i];
			state.v[0] = v0[i];
			state.v[1] = v1[i];
			state.v[2] = v2[i];
			state.v[3] = v3[i];
			state.v[4] = v4[i];
			state.boxmuller_flag = boxmuller_flag[i];
			state.boxmuller_flag_double = boxmuller_flag_double[i];
			state.boxmuller_extra = boxmuller_extra[i];
			state.boxmuller_extra_double = boxmuller_extra_double[i];
		}

		// Storing the states of the random number generator
		__device__ void store(const curandState_t &state, const unsigned int i)
		{
			d[i] = state.d;
			v0[i] = state.v[0];
			v1[i] = state.v[1];
			v2[i] = state.v[2];
			v3[i] = state.v[3];
			v4[i] = state.v[4];
			boxmuller_flag[i] = state.boxmuller_flag;
			boxmuller_flag_double[i] = state.boxmuller_flag_double;
			boxmuller_extra[i] = state.boxmuller_extra;
			boxmuller_extra_double[i] = state.boxmuller_extra_double;
		}

		// Initializing the random number generator
		__device__ void initialize(curandState_t &state, const unsigned int i, const unsigned int seed)
		{
			curand_init(seed,i,0,&state);
		}

		// Destructor
		~curandInitializer()
		{
			if(isCopy == false)
			{
				cudaFree(d);
				cudaFree(v0);
				cudaFree(v1);
				cudaFree(v2);
				cudaFree(v3);
				cudaFree(v4);
				cudaFree(boxmuller_flag);
				cudaFree(boxmuller_flag_double);
				cudaFree(boxmuller_extra);
				cudaFree(boxmuller_extra_double);
			}
		}

	// Private variables
	private:
		unsigned int *d, *v0, *v1, *v2, *v3, *v4;
  		int *boxmuller_flag, *boxmuller_flag_double;
  		float* boxmuller_extra;
  		double* boxmuller_extra_double;
		unsigned int N;
		bool isCopy;
};

// Initialize function for random number generators (for each thread)
__global__ void initialize_RNGs(curandInitializer initRNG, unsigned int clck)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	curandState_t state;

	initRNG.initialize(state,i,clck);
	initRNG.store(state,i);
}

// Initialize function for random number generators (for each warp)
__global__ void initialize_RNGs(curandInitializer initRNG, unsigned int clck,unsigned int WS)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	curandState_t state;

	initRNG.initialize(state,i/WS,clck);
	initRNG.store(state,i);
}

#endif
