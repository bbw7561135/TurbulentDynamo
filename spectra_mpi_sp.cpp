///
///  Fourier spectra MPI version (single-precision) for FlashUG
///
///  written by Christoph Federrath, 2012-2016
///

#include "mpi.h" /// MPI lib
#include <iostream>
#include <iomanip> /// for io manipulations
#include <sstream> /// stringstream
#include <fstream> /// for filestream operation
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fftw3-mpi.h> /// Fast Fourier Transforms MPI version
#include "FlashUG_mpi.h" /// Flash Uniform Grid class

// constants
int NDIM = 3;
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const int FAILURE = 0;
static const int MAX_NUM_BINS = 10048;
static const double pi = 3.14159265358979323846;

// MPI stuff
int MyPE = 0, NPE = 1;

// for FFTW
fftwf_complex *fft_data_x, *fft_data_y, *fft_data_z;
fftwf_plan fft_plan_x, fft_plan_y, fft_plan_z;

// for output
vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

/// forward function
vector<int> GetMetaData(const string inputfile, const string datasetname);
vector<int> InitFFTW(const vector<int> nCells);
void ReadParallel(const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds);
void SwapMemOrder(float * const data, const vector<int> N);
vector<int> InitFFTW(const vector<int> nCells);
void AssignDataToFFTWContainer(const string type, const vector<int> N, const long ntot_local, float * const dens, 
	const float * const velx, const float * const vely, const float * const velz);
void ComputeSpectrum(const vector<int> Dim, const vector<int> MyInds, const bool decomposition);
void WriteOutAnalysedData(const string OutputFilename);
void Normalize(float * const data_array, const long n, const double norm);
double Mean(const float * const data, const long size);
void Window(float* const data_array, const int nx, const int ny, const int nz);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NPE);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
	if (MyPE==0) cout<<"=== spectra_mpi_sp === MPI num procs: "<<NPE<<endl;
	if ((argc < 2)||(argc > 2)) {
		if (MyPE==0) cout << "Usage: spectra_mpi_sp INPUTFILEBASE" << endl;
		MPI_Finalize(); exit(FAILURE);
	}

	string inputfile = argv[1];
	long starttime = time(NULL);

	/// get the dataset dimensions
	vector<int> N = GetMetaData(inputfile, "velx");
	
	// signal dimensionality of the imulation
	if (N[Z]==1) NDIM = 2;

	/// allocate FFTW containers and create FTTW plan
	vector<int> MyInds = InitFFTW(N);
	long ntot_local = MyInds[1]*N[Y]*N[Z];
	if (Debug) cout<<"["<<MyPE<<"] MyInds: "<<MyInds[0]<<" "<<MyInds[1]<<" ntot_local="<<ntot_local<<endl;

    /// parallelisation / decomposition check
    int wrong_decompostion = 0, wrong_decompostion_red = 0;
    if (MyInds[1] != 0) {
        if (N[X] % MyInds[1] != 0) wrong_decompostion = 1;
    } else {
        wrong_decompostion = 1;
    }
    MPI_Allreduce(&wrong_decompostion, &wrong_decompostion_red, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (wrong_decompostion_red > 0) {
        if (MyPE==0) cout<<"Error: Number of cores is not multiple of N[X]."<<endl;
        MPI_Finalize();
        return 0;
    }

	/// allocate
	float *dens = new float[ntot_local];
	float *velx = new float[ntot_local];
	float *vely = new float[ntot_local];
	float *velz = 0;
    if (NDIM==3) {
		velz = new float[ntot_local];
	}
	// new:
	float *magx = new float[ntot_local];
	float *magy = new float[ntot_local];
	float *magz = 0;
	if (NDIM==3) {
		magz = new float[ntot_local];
	}

	/// read data
	if (MyPE==0) cout<<"start reading data from disk..."<<endl;
	ReadParallel(inputfile, "dens", dens, MyInds); if (MyPE==0) cout<<"dens read."<<endl;
	ReadParallel(inputfile, "velx", velx, MyInds); if (MyPE==0) cout<<"velx read."<<endl;
	ReadParallel(inputfile, "vely", vely, MyInds); if (MyPE==0) cout<<"vely read."<<endl;
    if (NDIM==3) {
		ReadParallel(inputfile, "velz", velz, MyInds); if (MyPE==0) cout<<"velz read."<<endl;
	}
	// new:
	ReadParallel(inputfile, "magx", magx, MyInds); if (MyPE==0) cout<<"magx read."<<endl;
	ReadParallel(inputfile, "magy", magy, MyInds); if (MyPE==0) cout<<"magy read."<<endl;
    if (NDIM==3) {
		ReadParallel(inputfile, "magx", magz, MyInds); if (MyPE==0) cout<<"magz read."<<endl;
	}

	/// normalization
	double norm_dens = 1.0;
	double norm_vels = 1.0;
	double norm_mags = 1.0;
	if (MyPE==0) {
		cout << "main:  norm_dens = " << norm_dens << endl;
		cout << "main:  norm_vels = " << norm_vels << endl;
		cout << "main:  norm_mags = " << norm_mags << endl;
	}
	if (false) {
		Normalize(dens, ntot_local, norm_dens);
		Normalize(velx, ntot_local, norm_vels);
		Normalize(vely, ntot_local, norm_vels);
		if (NDIM==3) Normalize(velz, ntot_local, norm_vels);
		// new:
		Normalize(magx, ntot_local, norm_mags);
		Normalize(magy, ntot_local, norm_mags);
		if (NDIM==3) Normalize(magz, ntot_local, norm_mags);
	}

	long endtime = time(NULL);
	int duration = endtime-starttime, duration_red = 0;
	if (Debug) cout << "["<<MyPE<<"] ****************** Local time for reading data = "<<duration<<"s ******************" << endl;
	MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (MyPE==0) cout << "****************** Global time for reading data = "<<duration_red<<"s ******************" << endl;	

	string outfilename = "";

	/// vels spectrum
	AssignDataToFFTWContainer("vels", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, true); // decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_vels.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// new: mags spectrum
	AssignDataToFFTWContainer("mags", N, ntot_local, dens, magx, magy, magz);	
	ComputeSpectrum(N, MyInds, true); // decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_mags.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// rho3 spectrum
	AssignDataToFFTWContainer("rho3", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, true); // decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_rho3.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// rhov spectrum
	AssignDataToFFTWContainer("rhov", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, true); // decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_rhov.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// varrho spectrum
	AssignDataToFFTWContainer("varrho", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, false); // no decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_varrho.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// varlnrho spectrum
	AssignDataToFFTWContainer("varlnrho", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, false); // no decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_varlnrho.dat";
		WriteOutAnalysedData(outfilename);
	}

/*
	/// rho spectrum
	AssignDataToFFTWContainer("rho", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, false); // no decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_rho.dat";
		WriteOutAnalysedData(outfilename);
	}

	/// lnrho spectrum
	AssignDataToFFTWContainer("lnrho", N, ntot_local, dens, velx, vely, velz);	
	ComputeSpectrum(N, MyInds, false); // no decomposition
	if (MyPE==0) {
		outfilename = inputfile+"_spect_lnrho.dat";
		WriteOutAnalysedData(outfilename);
	}
*/
	
	/// deallocate and clean
	delete [] dens; ///// DEALLOCATE
	delete [] velx; ///// DEALLOCATE
	delete [] vely; ///// DEALLOCATE
    if (NDIM==3) {
		delete [] velz; ///// DEALLOCATE
	}
	// new:
	delete [] magx; ///// DEALLOCATE
	delete [] magy; ///// DEALLOCATE
    if (NDIM==3) {
		delete [] magz; ///// DEALLOCATE
	}
	
	fftwf_free(fft_data_x);
	fftwf_free(fft_data_y);
    if (NDIM==3) {
		fftwf_free(fft_data_z);
	}

	fftwf_destroy_plan(fft_plan_x);
	fftwf_destroy_plan(fft_plan_y);
    if (NDIM==3) {
		fftwf_destroy_plan(fft_plan_z);
	}
	fftwf_mpi_cleanup();

	endtime = time(NULL);
	duration = endtime-starttime; duration_red = 0;
	if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
	MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;	

	MPI_Finalize();
	return 0;
} // end main ==========================================================



/// GetMetaData ==========================================================
vector<int> GetMetaData(const string inputfile, const string datasetname)
{
	/// FLASH file meta data
	FlashUG ug = FlashUG(inputfile);
	
	int NBLK = ug.GetNumBlocks();
	vector<int> NB = ug.GetNumCellsInBlock();
	vector< vector <vector<double> > > BB = ug.GetBoundingBox();
	vector< vector<double> > MinMaxDomain = ug.GetMinMaxDomain();
	vector<double> L = ug.GetL();
	vector<double> D = ug.GetD();
	vector<int> N = ug.GetN();

	if (MyPE==0) {
		cout<<"num blocks = "<<NBLK<<endl;
		cout<<"num cells in block = "<<NB[X]<<" "<<NB[Y]<<" "<<NB[Z]<<endl;
		cout<<"min domain = "<<MinMaxDomain[X][0]<<" "<<MinMaxDomain[Y][0]<<" "<<MinMaxDomain[Z][0]<<endl;
		cout<<"max domain = "<<MinMaxDomain[X][1]<<" "<<MinMaxDomain[Y][1]<<" "<<MinMaxDomain[Z][1]<<endl;
		cout<<"length of domain = "<<L[X]<<" "<<L[Y]<<" "<<L[Z]<<endl;
		cout<<"num cells in domain = "<<N[X]<<" "<<N[Y]<<" "<<N[Z]<<endl;
		cout<<"cell width = "<<D[X]<<" "<<D[Y]<<" "<<D[Z]<<endl;
	}

	return N;
} /// ===============================================================


/// InitFFTW ==========================================================
vector<int> InitFFTW(const vector<int> nCells)
{
	const bool Debug = false;

	const ptrdiff_t N[3] = {nCells[X], nCells[Y], nCells[Z]};
	ptrdiff_t alloc_local = 0, local_n0 = 0, local_0_start = 0;

	fftwf_mpi_init();

	// get local data size and allocate
    if (NDIM==3) {
		alloc_local = fftwf_mpi_local_size_3d(N[X], N[Y], N[Z], MPI_COMM_WORLD, &local_n0, &local_0_start);
	}
    if (NDIM==2) {
		alloc_local = fftwf_mpi_local_size_2d(N[X], N[Y], MPI_COMM_WORLD, &local_n0, &local_0_start);
	}
	/// ALLOCATE
	if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_x..."<<endl;
	fft_data_x = fftwf_alloc_complex(alloc_local);
	if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_y..."<<endl;
	fft_data_y = fftwf_alloc_complex(alloc_local);
    if (NDIM==3) {
		if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_z..."<<endl;
		fft_data_z = fftwf_alloc_complex(alloc_local);
	}
	if (Debug) cout<<"["<<MyPE<<"] ...alloc done."<<endl;

	/// PLAN
    if (NDIM==3) {
		if (Debug) cout<<"["<<MyPE<<"] fft_plan_x..."<<endl;
		fft_plan_x = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
		if (Debug) cout<<"["<<MyPE<<"] fft_plan_y..."<<endl;
		fft_plan_y = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
		if (Debug) cout<<"["<<MyPE<<"] fft_plan_z..."<<endl;
		fft_plan_z = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_z, fft_data_z, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
	}
    if (NDIM==2) {
		if (Debug) cout<<"["<<MyPE<<"] fft_plan_x..."<<endl;
		fft_plan_x = fftwf_mpi_plan_dft_2d(N[X], N[Y], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
		if (Debug) cout<<"["<<MyPE<<"] fft_plan_y..."<<endl;
		fft_plan_y = fftwf_mpi_plan_dft_2d(N[X], N[Y], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
	}
	if (Debug) cout<<"["<<MyPE<<"] ...plans done."<<endl;

	vector<int> ReturnVector(2);
	ReturnVector[0] = local_0_start;
	ReturnVector[1] = local_n0;
	
	return ReturnVector;
} /// ================================================================


/// ReadParallel ===============================================================
void ReadParallel(const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds)
{
	const bool Debug = false;
	
	/// FLASH file meta data
	FlashUG ug = FlashUG(inputfile);
	
	int NBLK = ug.GetNumBlocks();
	vector<int> NB = ug.GetNumCellsInBlock();
	vector< vector <vector<double> > > BB = ug.GetBoundingBox();
	vector< vector<double> > MinMaxDomain = ug.GetMinMaxDomain();
	vector<double> L = ug.GetL();
	vector<double> D = ug.GetD();
	vector<int> N = ug.GetN();

	/// find local affected blocks for this CPU
	vector<int> MyBlocks(0);
	double xl = (double)(MyInds[0])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
	double xr = (double)(MyInds[0]+MyInds[1])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
	if (Debug) cout<<"["<<MyPE<<"] xl= "<<xl<<" xr="<<xr<<endl;
	for (int b=0; b<NBLK; b++) {
		if ( ((xl <= BB[b][X][0]) && (BB[b][X][0] <  xr)) || ((xl <  BB[b][X][1]) && (BB[b][X][1] <= xr)) || ((xl >= BB[b][X][0]) && (BB[b][X][1] >= xr)) ) {
			MyBlocks.push_back(b);
		}
	}
	if (Debug) {
		cout<<"["<<MyPE<<"] My blocks = ";
		for (unsigned int b=0; b<MyBlocks.size(); b++) {
			cout<<MyBlocks[b]<<" "; cout << endl;
		}
	}

	double block_sum = 0.0;
	for (unsigned int ib=0; ib<MyBlocks.size(); ib++) {
		int b = MyBlocks[ib];
		vector<int> ind_xl = ug.CellIndexBlock(b, xl+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
		vector<int> ind_xr = ug.CellIndexBlock(b, xr-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
		vector<int> ind_bl = ug.CellIndexBlock(b, BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
		vector<int> ind_br = ug.CellIndexBlock(b, BB[b][X][1]-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
		vector<int> ind_dl = ug.CellIndexDomain(BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
		if (Debug) cout<<"ind_dl="<<ind_dl[X]<<" "<<ind_dl[Y]<<" "<<ind_dl[Z]<<endl;

		int is = ind_bl[X];
		if (ind_xl[X] > is) is = ind_xl[X];
		int ie = ind_br[X];
		if (ind_xr[X] < ie) ie = ind_xr[X];

	  	float *block_data = ug.ReadBlockVar(b, datasetname, MPI_COMM_NULL);

		if (Debug) {
			double sum = 0.0;
			for (long n = 0; n < NB[Z]*NB[Y]*NB[X]; n++) sum += block_data[n];
			block_sum += sum;

			long s_ind_data = (ind_dl[Z]+0)*N[Y]*MyInds[1] + (ind_dl[Y]+0)*MyInds[1] + (ind_dl[X]-MyInds[0]+is);
			long e_ind_data = (ind_dl[Z]+NB[Z]-1)*N[Y]*MyInds[1] + (ind_dl[Y]+NB[Y]-1)*MyInds[1] + (ind_dl[X]-MyInds[0]+ie);
			long s_ind_block = NB[X]*NB[Y]*0 + NB[X]*0 + is;
			long e_ind_block = NB[X]*NB[Y]*(NB[Z]-1) + NB[X]*(NB[Y]-1) + ie;
			cout<<"["<<MyPE<<"] (block, s_ind_block, e_ind_block, s_ind_data, e_ind_data)="
			<<b<<", "<<s_ind_block<<", "<<e_ind_block<<", "<<s_ind_data<<", "<<e_ind_data<<endl;
		}

		for (int k=0; k<NB[Z]; k++) {
			for (int j=0; j<NB[Y]; j++) {
				for (int i=is; i<=ie; i++) {
					long data_index = (ind_dl[Z]+k)*N[Y]*MyInds[1] + (ind_dl[Y]+j)*MyInds[1] + (ind_dl[X]-MyInds[0]+i);
					long block_index = k*NB[X]*NB[Y] + j*NB[X] + i;
					data_ptr[data_index] = block_data[block_index];
				}
			}
		}
		delete [] block_data;
	} // ib

	if (Debug) {
		double sum = 0.0, sum_red = 0.0;
		for (long n = 0; n < MyInds[1]*N[Y]*N[Z]; n++) sum += data_ptr[n];
		MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (MyPE==0) cout<<"before SwapMemOrder: sum(data_ptr of "+datasetname+")="<<sum_red<<endl;
	}

	/// SwapMemOrder
	vector<int> Dims(3); Dims[0] = MyInds[1]; Dims[1] = N[Y]; Dims[2] = N[Z];
	SwapMemOrder(data_ptr, Dims);

	if (Debug) {
		double sum = 0.0, sum_red = 0.0;
		for (long n = 0; n < MyInds[1]*N[Y]*N[Z]; n++) sum += data_ptr[n];
		MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (MyPE==0) cout<<"after SwapMemOrder: sum(data_ptr of "+datasetname+")="<<sum_red<<endl;
	}
} /// ==========================================================================


/// SwapMemOrder =====================================================
void SwapMemOrder(float * const data, const vector<int> N)
{
	const long ntot = N[0]*N[1]*N[2];
	float *tmp = new float[ntot];
	for (int i=0; i<N[0]; i++) for (int j=0; j<N[1]; j++) for (int k=0; k<N[2]; k++) {
		long ind1 = i*N[1]*N[2] + j*N[2] + k;
		long ind2 = k*N[1]*N[0] + j*N[0] + i;
		tmp[ind1] = data[ind2];
	}
	for (long i=0; i<ntot; i++) data[i] = tmp[i];
	delete [] tmp;
} /// ==============================================================


void AssignDataToFFTWContainer(const string type, const vector<int> N, const long ntot_local, float * const dens, const float * const velx, const float * const vely, const float * const velz)
{
	const double onethird = 1./3.;
	
	if (type=="rho") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = sqrt(dens[n]); /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = 0.; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = 0.; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="lnrho") {
		double mean_dens = Mean(dens, ntot_local);
		if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean dens = "<<mean_dens<<endl;
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = log(dens[n]/mean_dens); /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = 0.; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = 0.; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="varrho") {
		double mean_dens = Mean(dens, ntot_local);
		if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean dens = "<<mean_dens<<endl;
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = dens[n]-mean_dens; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = 0.; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = 0.; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="varlnrho") {
		for (long n=0; n<ntot_local; n++) dens[n] = log(dens[n]);
		double mean_lndens = Mean(dens, ntot_local);
		if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean log(dens) = "<<mean_lndens<<endl;
		for (long n=0; n<ntot_local; n++) dens[n] = exp(dens[n]);
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = log(dens[n])-mean_lndens; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = 0.; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = 0.; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="sqrtrho") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = sqrt(dens[n])*velx[n]; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = sqrt(dens[n])*vely[n]; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = sqrt(dens[n])*velz[n]; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="rho3") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = pow(dens[n],onethird)*velx[n]; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = pow(dens[n],onethird)*vely[n]; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = pow(dens[n],onethird)*velz[n]; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="rhov") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = dens[n]*velx[n]; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = dens[n]*vely[n]; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = dens[n]*velz[n]; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="vels") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = velx[n]; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = vely[n]; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = velz[n]; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	} else if (type=="mags") {
		for (long n=0; n<ntot_local; n++) {
			fft_data_x[n][0] = velx[n]; /// Real part
			fft_data_x[n][1] = 0.; /// Imaginary part
			fft_data_y[n][0] = vely[n]; /// Real part
			fft_data_y[n][1] = 0.; /// Imaginary part
			if (NDIM==3) {
				fft_data_z[n][0] = velz[n]; /// Real part
				fft_data_z[n][1] = 0.; /// Imaginary part
			}
		}
	}
	
	/// error check
	double sum = 0.0, sum_red = 0.0;
	double ntot = (double)(N[X])*(double)(N[Y])*(double)(N[Z]);
	if (NDIM==3) {
		for (long n = 0; n < ntot_local; n++) {
			sum += fft_data_x[n][0]*fft_data_x[n][0]+fft_data_y[n][0]*fft_data_y[n][0]+fft_data_z[n][0]*fft_data_z[n][0];
		}
	}
	if (NDIM==2) {
		for (long n = 0; n < ntot_local; n++) {
			sum += fft_data_x[n][0]*fft_data_x[n][0]+fft_data_y[n][0]*fft_data_y[n][0];
		}
	}
	if (Debug) cout << "["<<MyPE<<"] Local sum in physical space = " << sum/ntot << endl;
	MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (MyPE==0) cout << "Global sum in physical space ("<<type<<") = " << sum_red/ntot << endl;
} /// ==============================================================


/** --------------------- ComputeSpectrum ----------------------------
 **  computes total, transversal, longitudinal spectrum functions
 ** ------------------------------------------------------------------ */
void ComputeSpectrum(const vector<int> Dim, const vector<int> MyInds, const bool decomposition)
{
	const bool Debug = false;
	long starttime = time(NULL);
	
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: entering." << endl;
	
	if (NDIM==3) {
		if ((Dim[X]!=Dim[Y])||(Dim[X]!=Dim[Z])||(Dim[Y]!=Dim[Z])) {
			cout << "Spectra can only be obtained from cubic datasets (Nx=Ny=Nz)." << endl;
			exit(FAILURE);
		}
	}
	if (NDIM==2) {
		if (Dim[X]!=Dim[Y]) {
			cout << "Spectra can only be obtained from quadratic datasets (Nx=Ny)." << endl;
			exit(FAILURE);
		}
	}

	/////////// EXECUTE PLAN
	if (decomposition) {
		fftwf_execute(fft_plan_x);
		fftwf_execute(fft_plan_y);
		if (NDIM==3) {
			fftwf_execute(fft_plan_z);
		}
	} else {
		fftwf_execute(fft_plan_x);		
	}
	
	/// general constants
	const long N = Dim[X]; /// assume a cubic (square in 2D) box !
	long LocalNumberOfDataPoints = 0;
	long TotalNumberOfDataPoints = 0;
	if (NDIM==3) {
		LocalNumberOfDataPoints = N*N*MyInds[1];	
		TotalNumberOfDataPoints = N*N*N;
	}
	if (NDIM==2) {
		LocalNumberOfDataPoints = N*MyInds[1];	
		TotalNumberOfDataPoints = N*N;
	}
	const double TotalNumberOfDataPointsDouble = (double)(TotalNumberOfDataPoints);
	const double sqrt_TotalNumberOfDataPoints = sqrt((double)(TotalNumberOfDataPoints));

	/// allocate containers
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Allocating energy_spect..." << endl;
	double *energy_spect = new double[LocalNumberOfDataPoints];
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Allocating energy_lt_spect..." << endl;
	double *energy_lt_spect = new double[LocalNumberOfDataPoints];
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: ...allocating done." << endl;	
	for (long n = 0; n < LocalNumberOfDataPoints; n++) {
		energy_spect[n]    = 0.0;
		energy_lt_spect[n] = 0.0;
	}

	/// FFTW normalization
	if (decomposition) {
		if (NDIM==3) {
			for (long n = 0; n < LocalNumberOfDataPoints; n++) {
				fft_data_x[n][0] /= sqrt_TotalNumberOfDataPoints;
				fft_data_x[n][1] /= sqrt_TotalNumberOfDataPoints;
				fft_data_y[n][0] /= sqrt_TotalNumberOfDataPoints;
				fft_data_y[n][1] /= sqrt_TotalNumberOfDataPoints;
				fft_data_z[n][0] /= sqrt_TotalNumberOfDataPoints;
				fft_data_z[n][1] /= sqrt_TotalNumberOfDataPoints;
				energy_spect[n] += ( fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1] +
									fft_data_y[n][0]*fft_data_y[n][0]+fft_data_y[n][1]*fft_data_y[n][1] +
									fft_data_z[n][0]*fft_data_z[n][0]+fft_data_z[n][1]*fft_data_z[n][1] ) / TotalNumberOfDataPointsDouble;
			}
		}
		if (NDIM==2) {
			for (long n = 0; n < LocalNumberOfDataPoints; n++) {
				fft_data_x[n][0] /= sqrt_TotalNumberOfDataPoints;
				fft_data_x[n][1] /= sqrt_TotalNumberOfDataPoints;
				fft_data_y[n][0] /= sqrt_TotalNumberOfDataPoints;
				fft_data_y[n][1] /= sqrt_TotalNumberOfDataPoints;
				energy_spect[n] += ( fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1] +
									fft_data_y[n][0]*fft_data_y[n][0]+fft_data_y[n][1]*fft_data_y[n][1] ) / TotalNumberOfDataPointsDouble;
			}
		}
	} else {
		for (long n = 0; n < LocalNumberOfDataPoints; n++) {
			fft_data_x[n][0] /= sqrt_TotalNumberOfDataPoints;
			fft_data_x[n][1] /= sqrt_TotalNumberOfDataPoints;
			energy_spect[n] += (fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1]) / TotalNumberOfDataPointsDouble;
		}
	}
	
	double tot_energy_spect = 0.0, tot_energy_spect_red = 0.0;
	for (long n = 0; n < LocalNumberOfDataPoints; n++){
		tot_energy_spect += energy_spect[n];
	}
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local sum in spectral space = " << tot_energy_spect << endl;
	MPI_Allreduce(&tot_energy_spect, &tot_energy_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	tot_energy_spect = tot_energy_spect_red;
	if (MyPE==0) cout << "ComputeSpectrum: Global sum in spectral space = " << tot_energy_spect << endl;

	/// compute longitudinal spectrum (remember how FFTW sorts the k-values)
	double tot_energy_lt_spect = 0.0, tot_energy_lt_spect_red = 0.0;
	double dec_lt_0 = 0.0; double dec_lt_1 = 0.0;
	if (decomposition) {
		int k1 = 0; int k2 = 0; int k3 = 0;
        for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) { // parallelized bit (only loop local part)
			if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
			for (int l = 0; l < Dim[Y]; l++) {
				if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
				for (int m = 0; m < Dim[Z]; m++) {
					if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];
					double k_sqr_index = k1*k1 + k2*k2 + k3*k3;
			  		long index = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major
					if (NDIM==3) {
						dec_lt_0 = k1*fft_data_x[index][0] + k2*fft_data_y[index][0] + k3*fft_data_z[index][0];
						dec_lt_1 = k1*fft_data_x[index][1] + k2*fft_data_y[index][1] + k3*fft_data_z[index][1];
					}
					if (NDIM==2) {
						dec_lt_0 = k1*fft_data_x[index][0] + k2*fft_data_y[index][0];
						dec_lt_1 = k1*fft_data_x[index][1] + k2*fft_data_y[index][1];
					}
					if (k_sqr_index > 0) {
						energy_lt_spect[index] = (dec_lt_0*dec_lt_0+dec_lt_1*dec_lt_1)/k_sqr_index/TotalNumberOfDataPointsDouble;
					}
				}
			}
		}

		for (long n = 0; n < LocalNumberOfDataPoints; n++) tot_energy_lt_spect += energy_lt_spect[n];
		if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local sum of longitudinal part in spectral space = " << tot_energy_lt_spect << endl;
		MPI_Allreduce(&tot_energy_lt_spect, &tot_energy_lt_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		tot_energy_lt_spect = tot_energy_lt_spect_red;
		if (MyPE==0) cout << "ComputeSpectrum: Global sum of longitudinal part in spectral space = " << tot_energy_lt_spect << endl;
	} // decomposition

	/// compute the maximum k and construct the spect_grid i.e. the k_axis as well as a staggered grid
	const int    k_cut = N/2;
	const double log_increment_k = 0.01;
	const double increase_k_factor = pow(10.0, 2.0*log_increment_k);
	double       spect_grid_stag[MAX_NUM_BINS]; spect_grid_stag[0] = 0.0;
	double       spect_grid     [MAX_NUM_BINS]; spect_grid[0]      = 0.0;
	double       k_sqr = 1.0;
	int          bin_index = 0;
	while (k_sqr <= (k_cut+1.0)*(k_cut+1.0)) {
		bin_index++;
		if (k_sqr <= (k_cut+1.0)*(k_cut+1.0)) {
			k_sqr = bin_index*bin_index;
		} else {
		  k_sqr = k_sqr * increase_k_factor;
		}
		spect_grid_stag[bin_index] = k_sqr;
		if (bin_index >= MAX_NUM_BINS) {
			cout << "["<<MyPE<<"] ComputeSpectrum: ERROR. Number of spectral bins exceeds maximum number." << endl;
			exit(FAILURE);
		}
	}
	const int numbins = bin_index;

	/// construct the spectral grid
	for (int bin = 1; bin <= numbins; bin++){
		spect_grid[bin-1] = pow(sqrt(spect_grid_stag[bin-1])+(sqrt(spect_grid_stag[bin])-sqrt(spect_grid_stag[bin-1]))/2.0, 2.0);
	}

	/// calculate spectral densities
	if (MyPE==0 && Debug) cout << "ComputeSpectrum: Calculating spectral densities ..." << endl;
	double spect_binsum            [3][numbins]; /// contains longit., transv., total spectral densities
	double spect_binsum_sqr        [3][numbins]; /// this is used to compute the RMS and finally the sigma
	double sigma_spect_binsum      [3][numbins]; /// contains sigmas
	double spect_funct             [3][numbins]; /// contains longit., transv., total spectrum functions
	double sigma_spect_funct       [3][numbins]; /// contains sigmas
	double comp_lt_spect_funct        [numbins]; /// Kolmogorov compensated spectrum function
	double sigma_comp_lt_spect_funct  [numbins]; /// contains sigmas
	double comp_trsv_spect_funct      [numbins]; /// Kolmogorov compensated spectrum function
	double sigma_comp_trsv_spect_funct[numbins]; /// contains sigmas
	double diss_spect_funct           [numbins]; /// dissipative spectrum function
	double sigma_diss_spect_funct     [numbins]; /// contains sigmas
	double spect_binsum_lin           [numbins]; /// contains spectral densities of the non-decomposed dataset
	double spect_binsum_lin_sqr       [numbins]; /// this is used to compute the RMS and finally the sigma
	double sigma_spect_binsum_lin     [numbins]; /// contains sigmas
	double spect_funct_lin            [numbins]; /// contains lin spectrum function
	double sigma_spect_funct_lin      [numbins]; /// contains sigmas
	long   n_cells                    [numbins]; /// the number of cells inside a spherical shell in k-space
	long   n_count = 0;

	for (int bin = 0; bin < numbins; bin++) { /// set containers to zero
		if (decomposition) {
	    	for (int type = 0; type < 3; type++) { // type means long, trans, total
				spect_binsum      [type][bin] = 0.0;
				spect_binsum_sqr  [type][bin] = 0.0;
				sigma_spect_binsum[type][bin] = 0.0;
				spect_funct       [type][bin] = 0.0;
				sigma_spect_funct [type][bin] = 0.0;
			}
			comp_lt_spect_funct        [bin] = 0.0;
			sigma_comp_lt_spect_funct  [bin] = 0.0;
			comp_trsv_spect_funct      [bin] = 0.0;
			sigma_comp_trsv_spect_funct[bin] = 0.0;
			diss_spect_funct           [bin] = 0.0;
			sigma_diss_spect_funct     [bin] = 0.0;
		} else { // no decomposition
			spect_binsum_lin      [bin] = 0.0;
			spect_binsum_lin_sqr  [bin] = 0.0;
			sigma_spect_binsum_lin[bin] = 0.0;
			spect_funct_lin       [bin] = 0.0;
			sigma_spect_funct_lin [bin] = 0.0;
		}
		n_cells[bin] = 0;
	}

	int k1 = 0; int k2 = 0; int k3 = 0; // these are the time consuming loops (start optimization here)
    for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) { // the parallel bit
		if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
		for (int l = 0; l < Dim[Y]; l++) {
			if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
			for (int m = 0; m < Dim[Z]; m++) {
				if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];
				long k_sqr_index = k1*k1 + k2*k2 + k3*k3;
				int interval_l = 0; int interval_r = numbins-1; int bin_id = 0;
				while ((interval_r - interval_l) > 1) { /// nested intervals
					bin_id = interval_l + (interval_r - interval_l)/2;
					if (spect_grid[bin_id] > k_sqr_index) {
						interval_r = bin_id;
					} else {
						interval_l = bin_id;
					}
				}
				bin_id = interval_r;
				if ((bin_id <= 0) || (bin_id > numbins-1)) {
					cout << "["<<MyPE<<"] ComputeSpectrum: ERROR. illegal bin index." << endl;
					exit(FAILURE);
				}
				long index = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major
				if (decomposition) {
					double energy_trsv_spect = energy_spect[index] - energy_lt_spect[index];
					spect_binsum    [0][bin_id] += energy_lt_spect[index];
					spect_binsum    [1][bin_id] += energy_trsv_spect;
					spect_binsum    [2][bin_id] += energy_spect[index];
					spect_binsum_sqr[0][bin_id] += energy_lt_spect[index]*energy_lt_spect[index];
					spect_binsum_sqr[1][bin_id] += energy_trsv_spect*energy_trsv_spect;
					spect_binsum_sqr[2][bin_id] += energy_spect[index]*energy_spect[index];
				} else { // no decomposition
					spect_binsum_lin    [bin_id] += energy_spect[index];
					spect_binsum_lin_sqr[bin_id] += energy_spect[index]*energy_spect[index];
				}
				n_cells[bin_id]++;
				n_count++;
			} // m
	    } // l
	} // j

	/// resum the number of cells and total energy in k-space for error checking
	long n_cells_tot = 0, n_cells_tot_red = 0;
	tot_energy_spect = 0.0;
	for (int bin = 0; bin < numbins; bin++) {
		n_cells_tot += n_cells[bin];
		if (decomposition)  tot_energy_spect += spect_binsum[2][bin];
		if (!decomposition) tot_energy_spect += spect_binsum_lin[bin];
	}
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local ReSummed total number of cells   = " << n_cells_tot << endl;
	MPI_Allreduce(&n_cells_tot, &n_cells_tot_red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	n_cells_tot = n_cells_tot_red;
	if (MyPE==0) cout << "ComputeSpectrum: Global ReSummed total number of cells = " << n_cells_tot << endl;
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local ReSummed total in spectral space = " << tot_energy_spect << endl;
	MPI_Allreduce(&tot_energy_spect, &tot_energy_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	tot_energy_spect = tot_energy_spect_red;
	if (MyPE==0) cout << "ComputeSpectrum: Global ReSummed energy in spectral space = " << tot_energy_spect << endl;

	/// MPI Allreduce of the bin containers
	int *tmp_i_red = 0; double *tmp_d_red = 0;
	int *tmp_i = 0; double *tmp_d = 0;
	
	tmp_i_red = new int[numbins]; tmp_i = new int[numbins];
	for (int n=0; n<numbins; n++) tmp_i[n] = n_cells[n];
	MPI_Allreduce(tmp_i, tmp_i_red, numbins, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	for (int n=0; n<numbins; n++) n_cells[n]=tmp_i_red[n];
	delete [] tmp_i; delete [] tmp_i_red;

	tmp_d_red = new double[3*numbins]; tmp_d = new double[3*numbins];
	for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) tmp_d[3*n+dir] = spect_binsum[dir][n];
	MPI_Allreduce(tmp_d, tmp_d_red, 3*numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) spect_binsum[dir][n]=tmp_d_red[3*n+dir];
	for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) tmp_d[3*n+dir] = spect_binsum_sqr[dir][n];
	MPI_Allreduce(tmp_d, tmp_d_red, 3*numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) spect_binsum_sqr[dir][n]=tmp_d_red[3*n+dir];
	delete [] tmp_d; delete [] tmp_d_red;

	tmp_d_red = new double[numbins]; tmp_d = new double[numbins];
	for (int n=0; n<numbins; n++) tmp_d[n] = spect_binsum_lin[n];
	MPI_Allreduce(tmp_d, tmp_d_red, numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int n=0; n<numbins; n++) spect_binsum_lin[n]=tmp_d_red[n];
	for (int n=0; n<numbins; n++) tmp_d[n] = spect_binsum_lin_sqr[n];
	MPI_Allreduce(tmp_d, tmp_d_red, numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int n=0; n<numbins; n++) spect_binsum_lin_sqr[n]=tmp_d_red[n];
	delete [] tmp_d; delete [] tmp_d_red;
		
	/// write out (MASTER CPU only)
	if (MyPE==0) {
		/// calculate spectral densities and functions (normalization)
		for (int bin = 0; bin < numbins; bin++) {
			if (n_cells[bin] > 0) {
				if (decomposition) {
	      			for (int dir = 0; dir < 3; dir++) { /// long., transv., total
						spect_binsum      [dir][bin] /= static_cast<double>(n_cells[bin]);
						spect_binsum_sqr  [dir][bin] /= static_cast<double>(n_cells[bin]);
						sigma_spect_binsum[dir][bin]  = sqrt(spect_binsum_sqr[dir][bin] - spect_binsum[dir][bin]*spect_binsum[dir][bin]);
						if (NDIM==3) {
							spect_funct       [dir][bin]  = 4*pi*spect_grid_stag[bin]*spect_binsum      [dir][bin];
							sigma_spect_funct [dir][bin]  = 4*pi*spect_grid_stag[bin]*sigma_spect_binsum[dir][bin];
						}
						if (NDIM==2) {
							spect_funct       [dir][bin]  = 2*pi*sqrt(spect_grid_stag[bin])*spect_binsum      [dir][bin];
							sigma_spect_funct [dir][bin]  = 2*pi*sqrt(spect_grid_stag[bin])*sigma_spect_binsum[dir][bin];
						}
					}
					
	      			comp_lt_spect_funct        [bin] = pow(spect_grid_stag[bin], 2.0/2.0) * spect_funct      [0][bin];
	      			sigma_comp_lt_spect_funct  [bin] = pow(spect_grid_stag[bin], 2.0/2.0) * sigma_spect_funct[0][bin];
	      			comp_trsv_spect_funct      [bin] = pow(spect_grid_stag[bin], 5.0/6.0) * spect_funct      [1][bin];
	      			sigma_comp_trsv_spect_funct[bin] = pow(spect_grid_stag[bin], 5.0/6.0) * sigma_spect_funct[1][bin];
	      			diss_spect_funct           [bin] = spect_grid_stag[bin] * spect_funct      [2][bin];
	      			sigma_diss_spect_funct     [bin] = spect_grid_stag[bin] * sigma_spect_funct[2][bin];
				} else { // no decomposition
					spect_binsum_lin      [bin] /= static_cast<double>(n_cells[bin]);
					spect_binsum_lin_sqr  [bin] /= static_cast<double>(n_cells[bin]);
					sigma_spect_binsum_lin[bin]  = sqrt(spect_binsum_lin_sqr[bin] - spect_binsum_lin[bin]*spect_binsum_lin[bin]);
					if (NDIM==3) {
						spect_funct_lin       [bin]  = 4*pi*spect_grid_stag[bin]*spect_binsum_lin[bin];
						sigma_spect_funct_lin [bin]  = 4*pi*spect_grid_stag[bin]*sigma_spect_binsum_lin[bin];
					}
					if (NDIM==2) {
						spect_funct_lin       [bin]  = 2*pi*sqrt(spect_grid_stag[bin])*spect_binsum_lin[bin];
						sigma_spect_funct_lin [bin]  = 2*pi*sqrt(spect_grid_stag[bin])*sigma_spect_binsum_lin[bin];
					}
				}
			}
		}

		/// prepare OutputFileHeader
		OutputFileHeader.resize(0);
		stringstream dummystream;
		if (decomposition) {
			dummystream.precision(8);
			dummystream << "E_tot = " << endl;
			dummystream << scientific << tot_energy_spect << endl;
			dummystream << "E_lgt = " << endl;
			dummystream << scientific << tot_energy_lt_spect << endl << endl;
			dummystream << setw(30) << left << "#00_BinIndex";
			dummystream << setw(30) << left << "#01_KStag"              << setw(30) << left << "#02_K";
			dummystream << setw(30) << left << "#03_DK"                 << setw(30) << left << "#04_NCells";
			dummystream << setw(30) << left << "#05_SpectDensLgt"       << setw(30) << left << "#06_SpectDensLgtSigma";
			dummystream << setw(30) << left << "#07_SpectDensTrv"       << setw(30) << left << "#08_SpectDensTrvSigma";
			dummystream << setw(30) << left << "#09_SpectDensTot"       << setw(30) << left << "#10_SpectDensTotSigma";
			dummystream << setw(30) << left << "#11_SpectFunctLgt"      << setw(30) << left << "#12_SpectFunctLgtSigma";
			dummystream << setw(30) << left << "#13_SpectFunctTrv"      << setw(30) << left << "#14_SpectFunctTrvSigma";
			dummystream << setw(30) << left << "#15_SpectFunctTot"      << setw(30) << left << "#16_SpectFunctTotSigma";
			dummystream << setw(30) << left << "#17_CompSpectFunctLgt"  << setw(30) << left << "#18_CompSpectFunctLgtSigma";
			dummystream << setw(30) << left << "#19_CompSpectFunctTrv"  << setw(30) << left << "#20_CompSpectFunctTrvSigma";
			dummystream << setw(30) << left << "#21_DissSpectFunct"     << setw(30) << left << "#22_DissSpectFunctSigma";
			OutputFileHeader.push_back(dummystream.str()); dummystream.clear(); dummystream.str("");
		} else { // no decomposition
			dummystream << setw(30) << left << "#00_BinIndex";
			dummystream << setw(30) << left << "#01_KStag"           << setw(30) << left << "#02_K";
			dummystream << setw(30) << left << "#03_DK"              << setw(30) << left << "#04_NCells";
			dummystream << setw(30) << left << "#05_SpectDens"       << setw(30) << left << "#06_SpectDensSigma";
			dummystream << setw(30) << left << "#07_SpectFunct"      << setw(30) << left << "#08_SpectFunctSigma";
			OutputFileHeader.push_back(dummystream.str()); dummystream.clear(); dummystream.str("");
		}

		if (decomposition) {
	  		/// resize and fill WriteOutTable
	  		WriteOutTable.resize(numbins-2); /// spectrum output has numbins-2 lines
			for (unsigned int i = 0; i < WriteOutTable.size(); i++){
				WriteOutTable[i].resize(23); /// dec energy spectrum output has 23 columns
			}
			for (int bin = 1; bin < numbins-1; bin++) {
				int wob = bin-1;
				WriteOutTable[wob][ 0] = bin;
	      		WriteOutTable[wob][ 1] = sqrt(spect_grid_stag[bin]); /// k (staggered)
	      		WriteOutTable[wob][ 2] = sqrt(spect_grid     [bin]); /// k
	      		WriteOutTable[wob][ 3] = sqrt(spect_grid[bin])-sqrt(spect_grid[bin-1]); /// delta k
	      		WriteOutTable[wob][ 4] = n_cells                    [bin]; /// the number of cells in bin
	      		WriteOutTable[wob][ 5] = spect_binsum           [0] [bin]; /// longitudinal spectral density
	      		WriteOutTable[wob][ 6] = sigma_spect_binsum     [0] [bin]; /// sigma
	      		WriteOutTable[wob][ 7] = spect_binsum           [1] [bin]; /// transversal spectral density
	      		WriteOutTable[wob][ 8] = sigma_spect_binsum     [1] [bin]; /// sigma
	      		WriteOutTable[wob][ 9] = spect_binsum           [2] [bin]; /// total spectral density
	      		WriteOutTable[wob][10] = sigma_spect_binsum     [2] [bin]; /// sigma
	      		WriteOutTable[wob][11] = spect_funct            [0] [bin]; /// longitudinal spectrum function
	      		WriteOutTable[wob][12] = sigma_spect_funct      [0] [bin]; /// sigma
	      		WriteOutTable[wob][13] = spect_funct            [1] [bin]; /// transversal spectrum function
	      		WriteOutTable[wob][14] = sigma_spect_funct      [1] [bin]; /// sigma
	      		WriteOutTable[wob][15] = spect_funct            [2] [bin]; /// total spectrum function
	      		WriteOutTable[wob][16] = sigma_spect_funct      [2] [bin]; /// sigma
	      		WriteOutTable[wob][17] = comp_lt_spect_funct        [bin]; /// compensated longitudinal spectrum function
	      		WriteOutTable[wob][18] = sigma_comp_lt_spect_funct  [bin]; /// sigma
	      		WriteOutTable[wob][19] = comp_trsv_spect_funct      [bin]; /// compensated tranversal spectrum function
	      		WriteOutTable[wob][20] = sigma_comp_trsv_spect_funct[bin]; /// sigma
	      		WriteOutTable[wob][21] = diss_spect_funct           [bin]; /// dissipative spectrum function
	      		WriteOutTable[wob][22] = sigma_diss_spect_funct     [bin]; /// sigma
			}
		} else { // no decomposition
	  		/// resize and fill WriteOutTable
	  		WriteOutTable.resize(numbins-2); /// spectrum output has numbins-2 lines
			for (unsigned int i = 0; i < WriteOutTable.size(); i++) {
				WriteOutTable[i].resize(9); /// density spectrum output has 9 columns
			}
			for (int bin = 1; bin < numbins-1; bin++) {
				int wob = bin-1;
				WriteOutTable[wob][0] = bin;
	      		WriteOutTable[wob][1] = sqrt(spect_grid_stag[bin]); /// k (staggered)
	      		WriteOutTable[wob][2] = sqrt(spect_grid     [bin]); /// k
	      		WriteOutTable[wob][3] = sqrt(spect_grid[bin])-sqrt(spect_grid[bin-1]); /// delta k
	      		WriteOutTable[wob][4] = n_cells                   [bin]; /// the number of cells in bin
	      		WriteOutTable[wob][5] = spect_binsum_lin          [bin]; /// spectral density of non-decomposed dataset
	      		WriteOutTable[wob][6] = sigma_spect_binsum_lin    [bin]; /// sigma
	      		WriteOutTable[wob][7] = spect_funct_lin           [bin]; /// spectrum function of non-decomposed dataset
	      		WriteOutTable[wob][8] = sigma_spect_funct_lin     [bin]; /// sigma
			}
		}
	} // MyPE==0

	/// clean up
	delete [] energy_spect;
	delete [] energy_lt_spect;

	long endtime = time(NULL);
	int duration = endtime-starttime, duration_red = 0;
	if (Debug) cout << "["<<MyPE<<"] ****************** Local elapsed time for spectrum function computation = "<<duration<<"s ******************" << endl;
	MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (MyPE==0) cout << "****************** Global elapsed time for spectrum function computation = "<<duration_red<<"s ******************" << endl;
	if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: exiting." << endl;
}
/// =======================================================================


/** --------------------- Normalize -----------------------------------------------
 ** divide by norm
 ** ------------------------------------------------------------------------------- */
void Normalize(float * const data_array, const long n, const double norm)
{
	for (long i = 0; i < n; i++) data_array[i] /= norm;
} /// =======================================================================


/** ----------------------------- Mean -------------------------------
 **  computes the mean of a pointer-array
 ** ------------------------------------------------------------------ */
double Mean(const float * const data, const long size)
{
	long local_size = size;
	long global_size = 0;
	double value = 0.0, value_red = 0.0;
	for (long n = 0; n < local_size; n++){
		value += data[n];
	}
	MPI_Allreduce(&value, &value_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	value_red /= static_cast<double>(global_size);
	return value_red;
} /// =======================================================================


/** ------------------------ Window -----------------------------------------------
 ** apply window function to data ("Hann" or "Hanning" window in 3D)
 ** ------------------------------------------------------------------------------- */
void Window(float* const data_array, const int nx, const int ny, const int nz)
{
	if ((nx!=ny)||(nx!=nz)) {
		cout << "Window: only works for nx=ny=nz." << endl;
		exit(FAILURE);
	}
	const long nxy = nx*ny;
	const double twopi = 2.*pi;
	const double L = (double)(nx);

	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				double dx = (double)(i)+0.5-((double)(nx)/2.);
				double dy = (double)(j)+0.5-((double)(ny)/2.);
				double dz = (double)(k)+0.5-((double)(nz)/2.);
				double r = sqrt(dx*dx+dy*dy+dz*dz);
				long index = k*nxy+j*nx+i;
				if (r < L/2.) {
					data_array[index] *= 0.5*(1.0+cos((twopi*r)/L));
				} else {
					data_array[index] = 0.;
				}
			} //i
		} //j
	} //k
} /// =======================================================================


/** -------------------- WriteOutAnalysedData ---------------------------------
 **  Writes out a variable table of data and a FileHeader to a specified file
 ** --------------------------------------------------------------------------- */
void WriteOutAnalysedData(const string OutputFilename)
{
	/// open output file
	ofstream Outputfile(OutputFilename.c_str());
	
	/// check for file
	if (!Outputfile) {
		cout << "WriteOutAnalysedData:  File system error. Could not create '" << OutputFilename.c_str() << "'."<< endl;
		exit (FAILURE);
	} else { /// write data to output file
		cout << "WriteOutAnalysedData:  Writing output file '" << OutputFilename.c_str() << "' ..." << endl;
		for (unsigned int row = 0; row < OutputFileHeader.size(); row++) {
			Outputfile << setw(61) << left << OutputFileHeader[row] << endl; /// header
			if (false && Debug) cout << setw(61) << left << OutputFileHeader[row] << endl;
		}
	    for (unsigned int row = 0; row < WriteOutTable.size(); row++) { /// data
			for (unsigned int col = 0; col < WriteOutTable[row].size(); col++) {
				Outputfile << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
				if (false && Debug) cout << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
			}
			Outputfile << endl; if (false && Debug) cout << endl;
		}

		Outputfile.close();
		Outputfile.clear();
		cout << "WriteOutAnalysedData:  done!" << endl;
	}
} /// =======================================================================
