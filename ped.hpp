/**
 *  @file
 *  @brief Header file for periodic eigendecomposition algorithm.
 */

/**
 * \mainpage Periodic Eigendecomposition
 *
 * \section sec_intro Introduction 
 * This package contains the source file of implementing periodic 
 * eigendecomposition. 
 *
 * Suppose there are M matrices \f$J_M, J_{M-1},\cdots, J_1\f$, each of
 * which has dimension [N, N]. We are interested in the eigenvalues and
 * eigenvectors of their products:
 * \f[
 *  J_M J_{M-1}\cdots J_1\,, \quad, J_1J_M\cdots,J_2\,,\quad
 *  J_2J_1J_M\cdots J_3\,,\quad \cdots
 * \f]
 * Note all of these products have same eigenvalues and their
 * eigenvectors are related by similarity transformation.
 * This package is designed to solve this problem.
 *
 * The basic idea is the periodic Schur decompositon:
 * \f[
 *  J_i = Q_i R_i Q_{i-1}^\top
 * \f]
 * Such that
 * \f[
 *  J_M J_{M-1} \cdots J_1 = Q_M R_M R_{M-1}\cdots R_1 Q_M^\top
 * \f]
 * 
 * There is only one class PED which has only
 * member functions, no member variables. For the detailed usage,
 * please go to the documentation of two functions PED::EigVals()
 * and PED::EigVecs().
 * 
 * \section sec_usage How to compile
 * This package is developed under template library
 * <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page"><b>Eigen</b></a>.
 * In order to use this package, make sure you have
 * [Eigen3.2](http://eigen.tuxfamily.org/) or above,
 * and your C++ compiler support C++11.
 *
 * For example, the command line compilation in linux is
 *
 * \code
 *  g++ ped.cc yourcode.cc -std=c++0x -O3 -I/path/to/eigen/header
 * \endcode
 *
 * \section sec_ack Acknowledgment
 * This is one project for my PhD study. I sincerely thank my adviser
 * <a href="https://www.physics.gatech.edu/user/predrag-cvitanovic">
 * Prof. Predrag Cvitanovic </a>
 * for his patient guidance.
 */	


/** @class PED
 *  @brief A class to calculate the periodic eigendecomposition of
 *         the product of a sequence of matrices.
 *  @note This class require C++11 support.
 */

#ifndef PED_H
#define PED_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>
#include <vector> 
#include <utility>
#include <tuple>
#include <iostream>


using std::vector;
using std::pair; using std::make_pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix2d; using Eigen::Vector2d;
using Eigen::Matrix2cd; using Eigen::Vector2cd;
using Eigen::MatrixXcd; using Eigen::VectorXcd;
using Eigen::Map; using Eigen::Ref;
using Eigen::EigenSolver;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Eigen::SparseLU; 
using Eigen::COLAMDOrdering;
using Eigen::HouseholderQR;
using Eigen::Upper;

/*============================================================ *
 *                 Class : periodic Eigendecomposition         *
 *============================================================ */
class PED{
  
public:
    PED() {}

    MatrixXd 
    EigVals(MatrixXd &J, const int MaxN  = 100,
	    const double tol = 1e-16 , bool Print = true);
    pair<MatrixXd, MatrixXd>
    EigVecs(MatrixXd &J, const int MaxN  = 100,
	    const double tol = 1e-16, bool Print = true, const int trunc = 0);
    std::tuple<MatrixXd, vector<int>, MatrixXd> 
    eigenvalues(MatrixXd &J, const int MaxN = 100,
		const double tol = 1e-16, bool Print = true);
    MatrixXd getE(const MatrixXd &R, const std::vector<int> complex_index);
    MatrixXd getVbyPSE(const MatrixXd &R, const MatrixXd &Q, 
		       const std::vector<int> complex_index,
		       bool Print);
    pair<MatrixXd, vector<int> >
    PerSchur(MatrixXd &J, const int MaxN = 100,
	     const double tol = 1e-16, bool Print = true);
    MatrixXd 
    HessTrian(MatrixXd &G);
    vector<int> 
    PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
	       const int MaxN, const double tol, bool Print);
    //protected:
    void Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C,
		const int k);
    void Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
		const Vector2d &v, const int k);
    void HouseHolder(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, const int k,
		     bool subDiag = false);
    void GivensOneIter(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
		       const int L, const int U);
    void GivensOneRound(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
			const int k);
    vector<int> 
    checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
		     const int U,const double tol);
    vector<int> 
    padEnds(const vector<int> &v, const int &left, const int &right);
    double 
    deltaMat2(const Matrix2d &A);
    Vector2d 
    vecMat2(const Matrix2d &A);
    pair<Vector2d, Matrix2d> 
    complexEigsMat2(const Matrix2d &A);
    int 
    sgn(const double &num);
    pair<double, int> 
    product1dDiag(const MatrixXd &J, const int k);
    pair<Matrix2d, double> 
    product2dDiag(const MatrixXd &J, const int k);
    vector<int> 
    realIndex(const vector<int> &complexIndex, const int N);
    vector<Triplet<double> >
    triDenseMat(const Ref<const MatrixXd> &A, const size_t M = 0, 
		const size_t N = 0);
    vector<Triplet<double> > 
    triDenseMatKron(const size_t I, const Ref<const MatrixXd> &A, 
		    const size_t M = 0, const size_t N = 0);
    vector<Triplet<double> > 
    triDiagMat(const size_t n, const double x, 
	       const size_t M = 0, const size_t N = 0 );
    vector<Triplet<double> > 
    triDiagMatKron(const size_t I, const Ref<const MatrixXd> &A,
		   const size_t M = 0, const size_t N = 0 );
    pair<SparseMatrix<double>, VectorXd> 
    PerSylvester(const MatrixXd &J, const int &P, 
		 const bool &isReal, const bool &Print);
    MatrixXd
    oneEigVec(const MatrixXd &J, const int &P, 
	      const bool &isReal, const bool &Print);
    void
    fixPhase(MatrixXd &EigVecs, const VectorXd &realComplexIndex);
    void 
    reverseOrder(MatrixXd &J);
    void 
    reverseOrderSize(MatrixXd &J);
    void 
    Trans(MatrixXd &J);
    vector<int> 
    truncVec(const vector<int> &v, const int trunc);
    
    std::pair<MatrixXd, MatrixXd>
    QR(const Ref<const MatrixXd> &A);
    std::tuple<MatrixXd, MatrixXd, MatrixXd, vector<int> >
    PowerIter(const Ref<const MatrixXd> &J, 
	      const Ref<const MatrixXd> &Q,
	      const bool onlyLastQ,
	      int maxit, double Qtol, bool Print,
	      int PrintFreqency);
    MatrixXd PowerEigE(const Ref<const MatrixXd> &J, 
		       const Ref<const MatrixXd> &Q0,
		       int maxit, double Qtol, bool Print,
		       int PrintFreqency);
    bool checkQconverge(const Ref<const MatrixXd> &D, double Qtol);
    std::vector<int>
    getCplPs(const Ref<const MatrixXd> D, double Qtol);
    template<class Sqr>
    std::tuple<MatrixXd, MatrixXd, MatrixXd, vector<int> >
    PowerIter0(Sqr &sqr, const Ref<const MatrixXd> &Q0, 
	       const bool onlyLastQ,
	       int maxit, double Qtol, bool Print, 
	       int PrintFreqency);
    template<class Sqr>
    MatrixXd PowerEigE0(Sqr &sqr, const Ref<const MatrixXd> &Q0, 
			int maxit, double Qtol, bool Print, 
			int PrintFreqency);
};


/**
 * @brief Power iteration to obtain quasi-upper triangular form
 *
 * For a sequence J = \f$[ J_m, J_{m_1}, ..., J_1] \f$ and an inital orthonormal
 * matrix \f$ Q_0 \f$,
 * we use QR decomposition \f$ J_i Q_{i-1} = Q_i R_i \f$,
 * so \f$ J Q_m = Q_m R_m...R_2R_1 \f$.
 * 
 * Template function sqr is use to perform this squential QR decompostion.
 * It takes 2 arguments: `sqr(Q0, onlyLastQ)`. 
 * * onlyLastQ == true,  It returns \f$ Q_m \f$ and \f$ [R_m,..., R_2, R_1] \f$
 * * onlyLastQ == false, It returns \f$ [Q_m, ..., Q_2, Q_1]\f$ and \f$ [R_m,..., R_2, R_1] \f$
 *  
 * @param[in] sqr                  funtion to perform sequence QR decomposition
 * @param[in] onlyLastQ            only the last Q is returned
 * @param[in] Q0                   initial orthonormal matrix
 * @param[in] maxit                maximal number of iterations
 * @param[in] Qtol                 tolerance for convergence
 * @param[in] Print                print info or not
 * @param[in] PrintFrequency       print frequency                
 * 
 * @return   [Q, R, D, cp].   D is the diagonal matrix. cp is the complex eigenvalue positions
 * @see PerSchur
 */
template<class Sqr>
std::tuple<MatrixXd, MatrixXd, MatrixXd, vector<int> >
PED::PowerIter0(Sqr &sqr, const Ref<const MatrixXd> &Q0, 
		const bool onlyLastQ,
		int maxit, double Qtol, bool Print, 
		int PrintFreqency){
    
    MatrixXd Q(Q0);
    int N = Q.rows();
    int M = Q.cols(); 
    
    for(size_t i = 0; i < maxit; i++){
	if(Print && i%PrintFreqency == 0) printf("** power iter i = %zd/%d **\n", i, maxit);
	
	auto qr = sqr(Q, onlyLastQ);
	MatrixXd &Qp = qr.first; 
	MatrixXd &R = qr.second;
	MatrixXd D = Q.transpose() * Qp.leftCols(M); // true for both onlyLastQ
	
	bool c = checkQconverge(D, Qtol);
	bool e = (i == maxit - 1);
	if( c || e ){
	    if(Print){
		if(c)  printf("Power iteration converges at : i = %zd\n", i);
		else fprintf(stderr, "Power iteration not converge at i = %d.\n", maxit);
	    }
	    std::vector<int> cp = getCplPs(D, Qtol);
	    Qp.leftCols(M) = Q;		        
	    R.leftCols(M) = D * R.leftCols(M); // apply D to R_m
	    return std::make_tuple(Qp, R, D, cp);
	}
	
	Q = Qp.leftCols(M); // this should be the last line, otherwise there is error.
    }
}

/** 
 * @brief use power iteration to obtain the eigenvalues
 *
 * @see PowerIter0(), getE() 
 */
template<class Sqr>
MatrixXd
PED::PowerEigE0(Sqr &sqr, const Ref<const MatrixXd> &Q0, 
		int maxit, double Qtol, bool Print, 
		int PrintFreqency){
    auto power = PowerIter0(sqr, Q0, true, maxit, Qtol, Print, PrintFreqency);
    return  getE(std::get<1>(power), std::get<3>(power));
}


#endif	/* PED_H */

