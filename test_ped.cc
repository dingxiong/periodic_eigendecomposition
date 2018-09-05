/* How to compile:
 * g++ test_ped.cc -L../../lib/ -I ../../include/ -I/usr/local/home/xiong/apps/eigen/include/eigen3 -lped -ldenseRoutines -std=c++11 -O3
 */
#include <iostream>
#include "ped.hpp"
#include "denseRoutines.hpp"
#include <Eigen/Dense>
#include <cstdlib>
#include <complex>
#include <ctime>
using namespace std;
using namespace Eigen;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

int main(){
    //----------------------------------------
    cout.precision(16);
    
    switch(16){
    
    case 1 : 
	{
	    // small test of HessTrian.
	    MatrixXd A, B, C;
	    A = MatrixXd::Random(4,4);
	    B = MatrixXd::Random(4,4);
	    C = MatrixXd::Random(4,4); 
	    //MatrixXd A(4,4), B(4,4), C(4,4);
	    //A << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
	    //B = A.array() + 2;
	    //C = B.array() + 16;
	    cout << A << endl << endl;
	    cout << B << endl << endl;
	    cout << C << endl << endl;
	    //cout << A*B*C << endl << endl;
  
	    PED ped;
	    MatrixXd G(4,12);
	    G << A, B, C;
	    MatrixXd Q = ped.HessTrian(G);

	    cout << G.leftCols(4) << endl << endl;
	    cout << G.middleCols(4,4) << endl << endl;
	    cout << G.rightCols(4) << endl << endl;
	    //cout << A*B*C << endl << endl;
  
	    break;
	}

    case 2 :
	{
	    /*  Systematic test of HessTrian()               * 
	     *  					       * 
	     *  Sample output:			       * 
	     *  1.9984e-15				       * 
	     *  7.99361e-15				       * 
	     *  					       * 
	     * real	0m8.252s			       * 
	     * user	0m7.984s			       * 
	     * sys	0m0.208s                               *    
	     */
	    const int N = 100; 
	    const int M = 1000;
	    MatrixXd J = MatrixXd::Random(N, M*N);
	    MatrixXd J0 = J;
	    PED ped;
	    MatrixXd Q = ped.HessTrian(J);
  
	    double TriErr = 0; // error of the triangular form
	    double QJQErr = 0; //error of the transform
	    for(size_t i = 0; i < M; i++){
		// should be strict lower.
		MatrixXd tmp;
		if(0 == i){
		    tmp = J.middleCols(i*N, N).bottomLeftCorner(N-1, N-1).triangularView<StrictlyLower>();

		}else{
		    tmp = J.middleCols(i*N, N).triangularView<StrictlyLower>(); 
		}
		TriErr = max(TriErr, tmp.cwiseAbs().maxCoeff() );
    
		// error of transform QJQ - J
		MatrixXd dif = Q.middleCols(i*N, N).transpose() * J0.middleCols(i*N, N) 
		    * Q.middleCols(((i+1)%M)*N, N) - J.middleCols(i*N, N);
		QJQErr = max(QJQErr, dif.cwiseAbs().maxCoeff() );
	    }
	    cout << TriErr << endl;
	    cout << QJQErr << endl;

	    break;
	}
    
    case 3 :
	{
	    /* small test of GivensOneRound/GivensOneIter/PeriodicQR */
	    MatrixXd A, B, C;
	    A = MatrixXd::Random(4,4);
	    B = MatrixXd::Random(4,4);
	    C = MatrixXd::Random(4,4); 

	    PED ped;
	    MatrixXd J(4,12);
	    J << A, B, C;
	    MatrixXd Q = ped.HessTrian(J);

	    cout << J.leftCols(4) << endl << endl;
	    cout << J.middleCols(4,4) << endl << endl;
	    cout << J.rightCols(4) << endl << endl;
  
	    Vector2d tmp = J.block(0, 0, 2, 1);
	    // ped.GivensOneRound(J, Q, tmp, 0);
	    // ped.GivensOneIter(J, Q, tmp, 0, 3);
	    // for(int i = 0; i < 100; i++) ped.GivensOneIter(J, Q, tmp, 0, 3);
	    ped.PeriodicQR(J, Q, 0, 3, 100, 1e-15, true);
      

	    cout << J.leftCols(4) << endl << endl;
	    cout << J.middleCols(4,4) << endl << endl;
	    cout << J.rightCols(4) << endl << endl;
        
	    break;
	}

    case 4 :
	{
	    /* small test of PerSchur */
	    MatrixXd A, B, C;
	    A = MatrixXd::Random(4,4);
	    B = MatrixXd::Random(4,4);
	    C = MatrixXd::Random(4,4); 

	    PED ped;
	    MatrixXd J(4,12);
	    J << A, B, C;
	    MatrixXd J0 = J;

	    cout << J.leftCols(4) << endl << endl;
	    cout << J.middleCols(4,4) << endl << endl;
	    cout << J.rightCols(4) << endl << endl;
  
	    pair<MatrixXd, vector<int> > tmp = ped.PerSchur(J, 100, 1e-15);
	    MatrixXd Q = tmp.first;
	    vector<int> cp = tmp. second;

	    cout << J.leftCols(4) << endl << endl;
	    cout << J.middleCols(4,4) << endl << endl;
	    cout << J.rightCols(4) << endl << endl;
	    cout << Q.leftCols(4).transpose() * J0.leftCols(4) * Q.middleCols(4,4) << endl << endl;
	    cout << J.leftCols(4).diagonal<-1>() << endl;
	    for(vector<int>::iterator it = cp.begin(); it != cp.end(); it++) cout << *it << endl;
	    break;
	}
    
    case 5 :
	{
	    /*    Systematic test of PerSchur()
	     *
	     * sample output:
	     *    TriErr = 2.12135e-15
	     *    QJQErr = 1.59872e-14
	     *
	     *    real	2m32.313s
	     *    user	2m31.293s
	     *    sys	0m0.228s
	     */

	    const int N = 100; 
	    const int M = 1000;
	    srand(time(NULL));
	    MatrixXd J = MatrixXd::Random(N, M*N);
	    MatrixXd J0 = J;
	    PED ped;
	    pair<MatrixXd, vector<int> > tmp = ped.PerSchur(J, 1000, 1e-17);
	    MatrixXd Q = tmp.first;
	    vector<int> cp = tmp.second;
	    for(vector<int>::iterator it = cp.begin(); it != cp.end(); it++) cout << *it << endl;
      
	    double TriErr = 0; // error of the triangular form
	    double QJQErr = 0; //error of the transform
	    for(size_t i = 0; i < M; i++){
		// should be strict lower.
		MatrixXd tmp;
		if(0 == i){
		    tmp = J.middleCols(i*N, N).bottomLeftCorner(N-1, N-1).triangularView<StrictlyLower>();

		}else{
		    tmp = J.middleCols(i*N, N).triangularView<StrictlyLower>(); 
		}
		TriErr = max(TriErr, tmp.cwiseAbs().maxCoeff() );
    
		// error of transform QJQ - J
		MatrixXd dif = Q.middleCols(i*N, N).transpose() * J0.middleCols(i*N, N) 
		    * Q.middleCols(((i+1)%M)*N, N) - J.middleCols(i*N, N);
		QJQErr = max(QJQErr, dif.cwiseAbs().maxCoeff() );
	    }
	    cout << "TriErr = " << TriErr << endl;
	    cout << "QJQErr = " << QJQErr << endl;
      
	    // print out the subdiagonal elements
	    VectorXd sd = J.leftCols(N).diagonal<-1>();
	    cout << sd << endl;
	    break;
	}
    
    case 6 : // test realIndex() function
	{
	    vector<int> a{0, 3, 7, 9};
	    PED ped;
	    vector<int> b = ped.realIndex(a, 14);
	    for(vector<int>::iterator it = b.begin(); it != b.end(); it++) 
		cout << *it << endl; 
      
	    break;
	}

    case 7 : //tese eigsMat2() function
	{
	    Matrix2d A;
	    A << 1, -1, 1, 1;
	    PED ped;
	    pair<Vector2d, Matrix2d> tmp = ped.complexEigsMat2(A); 
	    cout << tmp.first << endl;
	    cout << tmp.second << endl;
      
	    break;
	}

    case 8 : // small test EigVals() function
	{
	    const int N = 10;
	    const int M = 40;
	    srand(time(NULL));

	    PED ped;
	    MatrixXd J = MatrixXd::Random(N, M*N);
	    MatrixXd J0 = MatrixXd::Identity(N,N);
	    for(size_t i = 0; i < M; i++) J0 = J0*J.middleCols(i*N,N);
  
	    MatrixXd eigs = ped.EigVals(J, 5000, 1e-16, true);
	    cout << eigs << endl << endl;

	    EigenSolver<MatrixXd> eg(J0);
	    MatrixXd tmp(N, 2);
	    VectorXcd eigs2 = eg.eigenvalues();
	    cout << eigs2 << endl << endl;
	    tmp << eigs2.cwiseAbs().array().log(),  eigs2.imag().array() / eigs2.real().array() ;
	    cout << tmp << endl;
      
	    break;

	}

    case 9 : // test the sparse matrix function triDenseMat()
	{
	    srand(time(NULL));
	    MatrixXd A = MatrixXd::Random(4,3);
	    PED ped;
	    vector<Tri> nz = ped.triDenseMat(A, 2, 1);
	    SpMat B(8,8);
	    B.setFromTriplets(nz.begin(), nz.end());
	    cout << B << endl;
	    break;
	}

    case 10 : // test the sparse matrix function triDenseMatKron()
	{
	    srand(time(NULL));
	    MatrixXd A = MatrixXd::Random(3,2);
	    PED ped;
	    vector<Tri> nz = ped.triDenseMatKron(2, A, 2, 1);
	    SpMat B(10,10);
	    B.setFromTriplets(nz.begin(), nz.end());
	    cout << B << endl;
	    break;
	}

    case 11 : // test the sparse matrix function triDiagMat()
	{
	    srand(time(NULL));
	    MatrixXd A = MatrixXd::Random(3,2);
	    PED ped;
	    vector<Tri> nz = ped.triDiagMat(3, 1.2, 2, 1);
	    SpMat B(10,10);
	    B.setFromTriplets(nz.begin(), nz.end());
	    cout << B << endl;
	    break;
	}

    case 12 : // test the sparse matrix function triDiagMatKron()
	{
	    srand(time(NULL));
	    MatrixXd A = MatrixXd::Random(3,2);
	    PED ped;
	    vector<Tri> nz = ped.triDiagMatKron(3, A, 2, 1);
	    SpMat B(12,12);
	    B.setFromTriplets(nz.begin(), nz.end());
	    cout << MatrixXd(B) << endl;
	    break;
	}

    case 13 : // small test PerSylvester() function
	{
	    const int N = 5;
	    const int M = 3;
	    srand(time(NULL));

	    PED ped;
	    MatrixXd J = MatrixXd::Random(N, M*N);
	    cout << J << endl << endl;
	    pair<SpMat, VectorXd> tmp = ped.PerSylvester(J, 1, false, true);
      
 
	    cout << MatrixXd(tmp.first) << endl << endl;
	    cout << tmp.second << endl << endl;

	    break;

	}

    case 14 : // small test of EigVecs() function
	{
	    const int N = 5;
	    const int M = 3;
	    srand(time(NULL));

	    PED ped;
	    MatrixXd J = MatrixXd::Random(N, M*N);
	    for(size_t j = 0; j < M; j++){
		MatrixXd J0 = MatrixXd::Identity(N,N);
		for(size_t i = 0; i < M; i++) J0 = J0*J.middleCols( ((i+M-j)%M)*N, N );
		EigenSolver<MatrixXd> eg(J0);
		MatrixXd tmp(N, 2);
		VectorXcd eigs2 = eg.eigenvalues();
		tmp << eigs2.cwiseAbs().array().log(),  eigs2.imag().array() / eigs2.real().array() ;
		cout << tmp << endl;
		MatrixXcd vec = eg.eigenvectors();
		for(size_t i = 0; i < N; i++){ // make the first element of each vector to be real.
		    complex<double> a = vec(0,i);
		    vec.col(i) = vec.col(i).array() * conj(a)/abs(a);
		}
		cout << vec << endl;
	
	    }
	    // MatrixXd eigs = ped.EigVals(J, 5000, 1e-16, true);
	    pair<MatrixXd, MatrixXd> ve = ped.EigVecs(J, 1000, 1e-16, true);
	    MatrixXd &EigVals = ve.first;
	    MatrixXd &EigVecs = ve.second;
	    ped.fixPhase(EigVecs, EigVals.col(2));
	    cout << EigVals << endl << endl;
	    cout << EigVecs << endl << endl;
	    break;
 
	}
    
    case 15: // test Trans() function
	{
	    srand(time(NULL));
	    MatrixXd A, B, C, D;
	    A = MatrixXd::Random(4,4);
	    B = MatrixXd::Random(4,4);
	    C = MatrixXd::Random(4,4);
	    D = MatrixXd::Random(4,4); 

	    cout << A << endl << endl;
	    cout << B << endl << endl;
	    cout << C << endl << endl;
	    cout << D << endl << endl;

	    PED ped;
	    MatrixXd J(4,16);
	    J << A, B, C, D;
	    ped.Trans(J);

	    cout << J.leftCols(4) << endl << endl;
	    cout << J.middleCols(4,4) << endl << endl;
	    cout << J.middleCols(8,4) << endl << endl;
	    cout << J.rightCols(4) << endl << endl;
      
	    break;
	}

    case 16: {
      
	srand(time(NULL));
	MatrixXd J, Q;
	
	J = MatrixXd::Zero(3, 3);
	Q = MatrixXd::Random(3, 3);
	J(2, 0) = 1;
	J(1, 1) = 1;
	J(0, 2) = 1;
	
	PED ped;
	auto tmp = ped.PowerIter(J, Q, true, 1000, 1e-14, true, 100);
	cout << Q << endl << endl;
	cout << std::get<0>(tmp) << endl << endl;
	cout << std::get<1>(tmp) << endl << endl;
	cout << std::get<2>(tmp) << endl << endl;
	// cout << std::get<3>(tmp);
	
	MatrixXd &R = std::get<1>(tmp);
	MatrixXd &D = std::get<2>(tmp);
	cout << denseRoutines::eEig(R) << endl;
	cout << D.col(0).norm() << endl;
	cout << D.col(1).norm() << endl;
	cout << D.col(2).norm() << endl;
	
	break;
    }

    default : 
	cout << "please indicate the block to be tested!"<< endl;

    }


}
