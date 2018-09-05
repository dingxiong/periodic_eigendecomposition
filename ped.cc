/** @file
 *  @brief Source file for the periodic eigendecomposition algorithm.
 */

#include "ped.hpp"
#include <cmath>
#include <complex>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

using std::cout; using std::endl;
/*============================================================*
 *            Class : periodic Eigendecomposition             *
 *============================================================*/

/*--------------------  constructor, destructor --------------- */

/*---------------        member methods          --------------- */

/** @brief Eigvals() calculate the eigenvalues of the product of a sequence of
 *         matrices in the form of \f$ \exp(\mu+i\omega) \f$.
 *
 * It returns a
 * [N,3] matrix, with the first column  stores \f$ \mu \f$, the second
 * column stores
 * \f$ \pm 1\f$ for real eigenvalues or \f$ \omega  \f$ for complex
 * eigenvalues, and the third column states whether the eigenvalue
 * is real (0) or complex (1).
 *
 * Example usage:
 * \code
 * MatrixXd J3 = MatrixXd::Random(5,5);
 * MatrixXd J2 = MatrixXd::Random(5,5); 
 * MatrixXd J1 = MatrixXd::Random(5,5);
 * MatrixXd J(5,15);
 * J << J3, J2, J1;
 * PED ped;
 * MatrixXd eigvals = ped.Eigvals(J, 1000, 1e-16, false);
 * cout << eigvals << endl << endl;
 * \endcode
 * The above code return the eigenvalues of \f$ J_3J_2J1\f$ in the format of
 * \f$ \exp(\mu+i\omega)\f$.
 * 
 *  @param[in] MaxN Maximal number of periodic QR iteration.
 *  @param[in] J a sequence of matrices. Dimension [N, N*M].
 *  @param[in] tol Tolerance used to check the convergence of the iteration
 *                 process.
 *  @param[in] Print Indicate whether to print the intermediate information.
 *  @return Matrix of size [N,3]. 
 *
 */
MatrixXd PED::EigVals(MatrixXd &J, const int MaxN /* = 100 */,
		      const double tol/* = 1e-16 */, bool Print /* = true */){ 
    std::tuple<MatrixXd, vector<int>, MatrixXd> tmp = eigenvalues(J, MaxN, tol, Print);
  
    return std::get<0>(tmp);
}

/** @brief calculate the eigenvectors of the product of a sequence of matrices
 *  
 *  Suppose Input matrix \f$J\f$ is a sequence of matrices with the same dimension
 *  [N,N]: \f$[J_M, J_{M-1}, \cdots, J_1]\f$. Function PED::EigVecs() will return
 *  you two matrices.
 *
 *  The first one is the eigen-matrix, for the specific format, see
 *  PED::EigVals(). The second matrix has dimension
 *  [N*N, M]. The first column stores the eigenvectors of product
 *  \f$J_M J_{M-1}\cdots J_1\f$ stacked up-down 
 *  \f$[v_1, v_2,\cdots,v_M]^{\top}\f$. The second column stores the eigenvectors
 *  of product \f$J_1 J_{M}\cdots J_2\f$, and similarly, for the remaining
 *  columns. 
 *
 *  @note For complex eigenvectors, they always appear in complex conjugate pairs.
 *  So, the real and imaginary parts are stored separately. For example, if the
 *  \f$i_{th}\f$ and \f$(i+1)_{th}\f$ vectors are complex pair, then elements
 *  from \f$N(i-1)+1\f$ to \f$Ni\f$ is the real part, and elements from
 *  \f$Ni+1\f$ to \f$N(i+1)\f$ is the imaginary part. They are stored in this way
 *  for better usage of memory.
 *
 * Example usage:
 * \code
 * MatrixXd J3 = MatrixXd::Random(5,5);
 * MatrixXd J2 = MatrixXd::Random(5,5); 
 * MatrixXd J1 = MatrixXd::Random(5,5);
 * MatrixXd J(5,15);
 * J << J3, J2, J1;
 * PED ped;
 * std::pair<MatrixXd, MatrixXd> eigs = ped.EigVecs(J, 1000, 1e-16, false);
 * cout << eigs.first << endl << endl; // print eigenvalues
 * cout << eigs.second << endl << endl; // print eigenvectors.
 * \endcode
 * The above code return the eigenvalues (in the format of
 * \f$ \exp(\mu+i\omega)\f$) and
 * eigenvectors of \f$ J_3J_2J_1\f$, \f$ J_1J_3J_2\f$, \f$ J_2J_1J_3\f$. 
 * 
 *  @param[in] MaxN Maximal number of periodic QR iteration.
 *  @param[in] J a sequence of matrices. Dimension [N, N*M].
 *  @param[in] tol Tolerance used to check the convergence of the iteration
 *                 process.
 *  @param[in] Print Indicate whether to print the intermediate information.
 *  @param[in] trunc indicate the subset of eigenvectors to get
 *  @return a pair of matrices storing eigenvalues and eigenvectors. 
 */

pair<MatrixXd, MatrixXd> PED::EigVecs(MatrixXd &J, const int MaxN /* = 100 */,
				      const double tol /* = 1e-16 */, 
				      bool Print /* = true*/, 
				      const int trunc /* = 0 */){
    const int N = J.rows();
    const int M = J.cols() / N;
    int Trunc = trunc;
    if(Trunc == 0) Trunc = N; // set to full

    std::tuple<MatrixXd, vector<int>, MatrixXd> tmp = eigenvalues(J, MaxN, tol, Print);
    MatrixXd &Eig_vals = std::get<0>(tmp);
    vector<int> &complex_index = std::get<1>(tmp);
    MatrixXd &Q = std::get<2>(tmp);
    vector<int> real_index = realIndex(complex_index, N);

    // get trunction indices. Note the special treatment of complex indices.
    vector<int> real_index_trunc = truncVec(real_index, Trunc - 1);
    vector<int> complex_index_trunc = truncVec(complex_index, Trunc - 2);
  
    MatrixXd eigVecs(N*Trunc, M);
  
    // get the real eigenvectors
    for(vector<int>::iterator it = real_index_trunc.begin(); 
	it != real_index_trunc.end(); it++)
	{
	    MatrixXd vec = oneEigVec(J, *it, true, Print); 
	    for(size_t i = 0; i < M; i++){
		// note eigenvector of J1JmJm-1...J2 is related to that of Jm..J1 by Q1.
		// and Qi are stored in order Qm, Qm-1, ..., Q1
		// Ri are transformed by Qi-1
		eigVecs.block((*it)*N, i, N, 1) = Q.middleCols( ((M-i)%M)*N, N) * vec.col(i);
		double norm = eigVecs.block((*it)*N, i, N, 1).norm();
		eigVecs.block((*it)*N, i, N, 1) = eigVecs.block((*it)*N, i, N, 1).array()/norm; 
	    }
	}
  
    // get the complex eigenvectors
    for(vector<int>::iterator it = complex_index_trunc.begin(); 
	it != complex_index_trunc.end(); it++){
	MatrixXd vec = oneEigVec(J, *it, false, Print);
	for(size_t i = 0; i < M; i++){
	    eigVecs.block((*it)*N, i, N, 1) = Q.middleCols(((M-i)%M)*N, N) * vec.col(2*i);
	    eigVecs.block((*it+1)*N, i, N, 1) = Q.middleCols(((M-i)%M)*N, N) * vec.col(2*i+1);
	    double norm = sqrt( eigVecs.block((*it)*N, i, N, 1).squaredNorm() +
				eigVecs.block((*it+1)*N, i, N, 1).squaredNorm() );
	    eigVecs.block((*it)*N, i, N, 1) = eigVecs.block((*it)*N, i, N, 1).array() / norm;
	    eigVecs.block((*it+1)*N, i, N, 1) =  eigVecs.block((*it+1)*N, i, N, 1).array() / norm;
	}
    }
    return make_pair(Eig_vals, eigVecs);

}

/** @brief calculate the periodic Schur decomposition and the eigenvalues of the
 *         product of a sequence of matrices.
 *
 *  @return A tuple consists of eigenvalues, position of complex eigenvalues and transform
 *         matrices.
 *
 */
std::tuple<MatrixXd, vector<int>, MatrixXd> 
PED::eigenvalues(MatrixXd &J, const int MaxN /* = 100 */,
		 const double tol/* = 1e-16 */, bool Print /* = true */){ 
    const int N = J.rows();
    pair<MatrixXd, vector<int> > tmp = PerSchur(J, MaxN, tol, Print);
    MatrixXd &Q = tmp.first;
    vector<int> complex_index = tmp.second;
    MatrixXd eigVals = getE(J, complex_index);
      
    return std::make_tuple(eigVals, complex_index, Q);
  
}

/** 
 * @brief calculate eigenvalues given a sequence of (quasi-)upper triangular matrices
 *
 * @param[in] R                   a sequence of upper triangular matrices \f$ [R_m, ... R_2, R_1] \f$
 *                                among which \f$ R_m \f$ is quasi-upper triangular
 * @param[in] complex_index       indices of complex eigenvalue positions
 * @return   eigenvalues in form of [log, phase, complex indication]
 */
MatrixXd PED::getE(const MatrixXd &R, const std::vector<int> complex_index){
    const int N = R.rows();
    vector<int> real_index = realIndex(complex_index, N);
    MatrixXd eigVals(N, 3);
    
    // get the real eigenvalues
    for(auto it = real_index.begin(); it != real_index.end(); it++){
	pair<double, int> tmp = product1dDiag(R, *it);
	eigVals(*it, 0) = tmp.first;
	eigVals(*it, 1) = tmp.second;
	eigVals(*it, 2) = 0;
    }

    // get the complex eigenvalues
    for(auto it = complex_index.begin(); it != complex_index.end(); it++){
	pair<Matrix2d, double> tmp = product2dDiag(R, *it);
	pair<Vector2d, Matrix2d> tmp2 = complexEigsMat2(tmp.first);
    
	eigVals(*it, 0) = tmp.second + tmp2.first(0);
	eigVals(*it, 1) = tmp2.first(1);
	eigVals(*it, 2) = 1;
    
	eigVals(*it+1, 0) = tmp.second + tmp2.first(0);
	eigVals(*it+1, 1) = -tmp2.first(1);
	eigVals(*it+1, 2) = 1;
    }

  
    return eigVals;
  
}

/**
 * @brief use the Periodic Sylvester Equaution to obtain all eigenvectors simultaneously
 *
 * Note, R has dimension [N2, N2xM], Q has dimension [N, N2*M].
 * N is the dimension of the system, N2 is the number of leading eigenvectors wanted.
 * N2 is smaller or equal to N.
 * 
 * Eigenvectors are stored in a [N*N2, M] matrix. Each column contains N2 eigenvectors
 * corresponding to one cyclic rotation of original sequence. Also, complex
 * eigenvectors are split into its real/imaginary parts. All eigenvectros are normalized.
 *
 * Also the indices in complex_index should be smaller than N2.
 * 
 * @param[in] R                quasi-upper triangular matrices [R_m, ..., R_2, R_1]
 * @param[in] Q                orthogonal matrices
 * @param[in] complex_index    indices of complex eigenvectors
 * @return    eigenvector 
 */
MatrixXd PED::getVbyPSE(const MatrixXd &R, const MatrixXd &Q, 
			const std::vector<int> complex_index,
			bool Print){
    assert(R.cols() == Q.cols());
    const int N = Q.rows();
    const int N2 = R.rows();
    const int M = R.cols() / N2;
    
    MatrixXd eigVecs(N*N2, M);
    
    std::vector<int> real_index = realIndex(complex_index, N2);
    
    // get the real eigenvectors
    for(auto it = real_index.begin(); it != real_index.end(); it++)
	{
	    MatrixXd vec = oneEigVec(R, *it, true, Print); 
	    for(size_t i = 0; i < M; i++){
		// note eigenvector of J1JmJm-1...J2 is related to that of Jm..J1 by Q1.
		// and Qi are stored in order Qm, Qm-1, ..., Q1
		// Ri are transformed by Qi-1
		eigVecs.block((*it)*N, i, N, 1) = Q.middleCols( ((M-i)%M)*N2, N2) * vec.col(i);
		double norm = eigVecs.block((*it)*N, i, N, 1).norm();
		eigVecs.block((*it)*N, i, N, 1) = eigVecs.block((*it)*N, i, N, 1).array()/norm; 
	    }
	}
  
    // get the complex eigenvectors
    for(auto it = complex_index.begin(); it != complex_index.end(); it++)
	{
	    MatrixXd vec = oneEigVec(R, *it, false, Print);
	    for(size_t i = 0; i < M; i++){
		eigVecs.block((*it)*N, i, N, 1) = Q.middleCols(((M-i)%M)*N2, N2) * vec.col(2*i);
		eigVecs.block((*it+1)*N, i, N, 1) = Q.middleCols(((M-i)%M)*N2, N2) * vec.col(2*i+1);
		double norm = sqrt( eigVecs.block((*it)*N, i, N, 1).squaredNorm() +
				    eigVecs.block((*it+1)*N, i, N, 1).squaredNorm() );
		eigVecs.block((*it)*N, i, N, 1) = eigVecs.block((*it)*N, i, N, 1).array() / norm;
		eigVecs.block((*it+1)*N, i, N, 1) =  eigVecs.block((*it+1)*N, i, N, 1).array() / norm;
	    }
	}
    return eigVecs;
}

/** @brief Periodic Schur decomposition of a sequence of matrix stored in J
 *
 *  @return pair value. First is the orthogonal transform matrix sequence Qm, Qm-1,..., Q1
 *          Second is the vector storing the positions of complex eigenvalues.
 */
pair<MatrixXd, vector<int> > PED::PerSchur(MatrixXd &J, const int MaxN /* = 100 */, 
					   const double tol/* = 1e-16*/, bool Print /* = true */){
    const int N = J.rows();
    MatrixXd Q = HessTrian(J);
    vector<int> cp = PeriodicQR(J, Q, 0, N-1, MaxN, tol, Print);
  
    return make_pair(Q, cp);
}

 
/* @brief transform the matrices stored in J into Hessenberg-upper-triangular form         * 
 * 											   * 
 * Input: J = [J_m, J_{m_1}, ..., J_1] , a sequence of matrices with each of which	   * 
 *        has dimension [n,n], so J has dimension [mn,n]. Note the storage is columnwise   * 
 *        We are interested in the product J_0 = J_m * J_{m-1} *,...,* J_1.		   * 
 * Output: J' = [J'_m, J'_{m_1}, ..., J'_1] in the Hessenberg upper-triangular form.	   * 
 *         J'_m: Hessenberg matrix; the others are upper-triangular.			   * 
 *         Q = [Q_m, Q_{m-1}, ..., Q_1], a sequence of orthogonal matrices, which satisfy  *   
 *         Q_i^{T} * J_i * Q_{i-1} = J'_i, with Q_0 = Q_m.                                 *
 *         
 * NOTE : THE TRANSFORM IS IN PLACE.  
 * */
MatrixXd PED::HessTrian(MatrixXd &J){
    const int N = J.rows();
    const int M = J.cols() / N;
    MatrixXd Q = (MatrixXd::Identity(N,N)).replicate(1, M);
  
    for(size_t i = 0; i < N - 1; i++){
	for(size_t j = M-1; j > 0; j--){
	    HouseHolder(J.middleCols((j-1)*N, N), J.middleCols(j*N, N), 
			Q.middleCols(j*N, N), i);
	}
	if(i < N - 2){
	    HouseHolder(J.middleCols((M-1)*N, N), J.middleCols(0, N), 
			Q.middleCols(0, N), i, true);
	}
    }
  
    return Q;
}

/**
 * PeriodicQR transforms an unreduced hessenberg-triangular sequence of
 * matrices into periodic Schur form.
 * This iteration method is based on the Implicit-Q theorem.
 *
 *
 */
vector<int> PED::PeriodicQR(MatrixXd &J, MatrixXd &Q, const int L, const int U,
			    const int MaxN, const double tol, bool Print){
    const int N = J.rows(); 
    const int M = J.cols() / N;
  
    switch(U - L){
    
	/* case 1: [1,1] matrix. No further operation needed.
	 *         Just return a empty vector.
	 */
    case 0 :
	{
	    if(Print) printf("1x1 matrix at L = U = %d \n", L);
	    return vector<int>();
	}

	/* case 2: [2,2] matrix. Need to determine whether complex or real
	 * if the eigenvalues are complex pairs,no further reduction is needed;
	 * otherwise, we need to turn it into diagonal form.
	 * */
    case 1 :
	{
	    Matrix2d mhess = MatrixXd::Identity(2,2);
	    // normalize the matrix to avoid overflow/downflow
	    for(size_t i = 0; i < M; i++) {
		mhess *= J.block<2,2>(L, i*N+L); // rows not change, cols increasing.
		mhess = mhess.array() / mhess.cwiseAbs().maxCoeff();
	    }
	    double del = deltaMat2(mhess);
	    // judge the eigenvalues are complex or real number.
	    if(del < 0){
		if(Print) printf("2x2 has complex eigenvalues: L = %d, U=%d\n", L, U);
		return vector<int>{L}; // nothing is needed to do.
	    }else{      
		Vector2d tmp = vecMat2(mhess);
		GivensOneRound(J, Q, tmp, L);
	    }
	    // no break here, since the second case need to use iteration too.
	}
  
    
	/* case 3 : subproblme dimension >= 3 or
	 *         real eigenvalues with dimension = 2.
	 */
    default : 
	{
	    vector<int> cp; // vector to store the position of complex eigenvalues.
	    size_t np;
	    for(np = 0; np < MaxN; np++){
		// Here we define the shift, but right now do not use any shift.
		Vector2d tmp = J.block(L, L, 2, 1);
		GivensOneIter(J, Q, tmp, L, U);
    
		vector<int> zeroID = checkSubdiagZero(J.leftCols(N), L, U, tol);
		vector<int> padZeroID = padEnds(zeroID, L-1, U); // pad L-1 and U at ends.
		const int Nz = padZeroID.size();
		if( Nz > 2 ){

		    //////////////////////////////////////////////////////
		    // print out divide conquer information.
		    if(Print) printf("subproblem L = %d, U = %d uses iteration %zd\n", L, U, np);
		    if(Print){
			printf("divide position: ");
			for(vector<int>::iterator it = zeroID.begin(); it != zeroID.end(); it++)
			    printf("%d ", *it);
			printf("\n");
		    }
		    /////////////////////////////////////////////////////

		    for(size_t i = 0; i < Nz-1; i++){ 	
			vector<int> tmp = PeriodicQR(J, Q, padZeroID[i]+1, padZeroID[i+1], MaxN, tol, Print);
			cp.insert(cp.end(), tmp.begin(), tmp.end());
		    }
		    break; // after each subproblem is finished, then problem is solved.
		}
    
	    }
	    ////////////////////////////////////////////////////////////
	    //print out the information if not converged.
	    if( np == MaxN){
		fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!\n");
		fprintf(stderr, "subproblem L = %d, U = %d does not converge in %d iterations!\n", L, U, MaxN);
		fprintf(stderr, "subdiagonal elements are :");
		for(size_t i = L; i < U; i++) fprintf(stderr, "%g ", J(i+1,i));
		fprintf(stderr, "\n!!!!!!!!!!!!!!!!!!!!!!!\n");
	    }
	    ////////////////////////////////////////////////////////////
	    return cp;
	}

    }
  
}


/* @brief perform Givens iteration across whole sequence of J.
 * 
 * */
void PED::GivensOneRound(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
			 const int k){
    const int N = J.rows(); 
    const int M = J.cols() / N;
  
    // Givens rotate the first matrix and the last matrix  
    Givens(J.rightCols(N), J.leftCols(N), Q.leftCols(N), v, k);
  
    // sequence of Givens rotations from the last matrix to the first matrix
    // the structure of first matrix is probabaly destroyed.
    for(size_t i = M-1; i > 0; i--)
	Givens(J.middleCols((i-1)*N, N), J.middleCols(i*N, N), Q.middleCols(i*N, N), k);
}

/* @brief perform periodic QR iteration and restore to the initial form
 * 
 * */
void PED::GivensOneIter(MatrixXd &J, MatrixXd &Q, const Vector2d &v, 
			const int L, const int U){
    // first Given sequence rotation specified by 2x1 vector v
    GivensOneRound(J, Q, v, L);

    // restore the Hessenberg upper triangular form by chasing down the bulge.
    for(size_t i = L; i < U-1; i++){ // cols from L to U-2
	Vector2d tmp = J.block(i+1, i, 2, 1); // take the subdiagonal 2 vector to form Givens.
	GivensOneRound(J, Q, tmp, i+1);
    }
}

/** @brief Givens rotation with provided 2x1 vector as parameter. 
 *
 */
void PED::Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
		 const Vector2d &v, const int k){
    double nor = v.norm();
    double c = v(0)/nor;
    double s = v(1)/nor;

    MatrixXd tmp = B.row(k) * c + B.row(k+1) * s;
    B.row(k+1) = -B.row(k) * s + B.row(k+1) * c;
    B.row(k) = tmp;
  
    MatrixXd tmp2 = A.col(k) * c + A.col(k+1) * s;
    A.col(k+1) = -A.col(k) * s + A.col(k+1) * c;
    A.col(k) = tmp2;

    MatrixXd tmp3 = C.col(k) * c + C.col(k+1) * s;
    C.col(k+1) = -C.col(k) * s + C.col(k+1) * c;
    C.col(k) = tmp3;
  
}

/* @brief insert Givens rotation between matrix product A*B.
 * A*G^{T}*G*B
 * G = [c s
 *     -s c]
 * G^{T} = [c -s
 *          s  c]
 * */
void PED::Givens(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C,
		 const int k){
    double nor = sqrt(B(k,k)*B(k,k) + B(k+1,k)*B(k+1,k)); 
    double c = B(k,k)/nor;
    double s = B(k+1,k)/nor;

    // rows k, k+1 of B are transformed
    MatrixXd tmp = B.row(k) * c + B.row(k+1) * s;
    B.row(k+1) = -B.row(k) * s + B.row(k+1) * c;
    B.row(k) = tmp;

    // columns k, k+1 of A are transformed
    MatrixXd tmp2 = A.col(k) * c + A.col(k+1) * s;
    A.col(k+1) = -A.col(k) * s + A.col(k+1) * c;
    A.col(k) = tmp2;
  
    MatrixXd tmp3 = C.col(k) * c + C.col(k+1) * s;
    C.col(k+1) = -C.col(k) * s + C.col(k+1) * c;
    C.col(k) = tmp3;
}

/* @brief store the position where the subdiagonal element is zero.
 *
 * The criteria : J(i+1, i) < 0.5 * tol * (|J(i,i)+J(i+1,i+1)|)
 * */
vector<int> PED::checkSubdiagZero(const Ref<const MatrixXd> &J0,  const int L,
				  const int U, const double tol){
    const int N = J0.rows();
    vector<int> zeroID; // vector to store zero position.
    zeroID.reserve(U - L);
    for(size_t i = L; i < U; i++){
	double SS = ( fabs(J0(i,i)) + fabs(J0(i+1,i+1)) ) * 0.5;
	if(fabs(J0(i+1, i)) < SS * tol) zeroID.push_back(i);
    }

    return zeroID;
}

/** @brief pad two elements at the begining and end of a vector.
 *
 */
vector<int> PED::padEnds(const vector<int> &v, const int &left, const int &right){
    vector<int> vp;
    vp.push_back(left); 
    vp.insert(vp.end(), v.begin(), v.end());
    vp.push_back(right);

    return vp;
}

/** @brief realIndex() gets the positions of the real eigenvelues from the positions
 *         of complex eigenvalues.
 *
 *  Denote the sequence of complex positions: [a_0, a_1,...a_s], then the real
 *  positions between [a_i, a_{i+1}] is from a_i + 2 to a_{i+1} - 1.
 *  Example:
 *          Complex positions :  3, 7, 9
 *          Dimension : N = 12
 *      ==> Real positions : 0, 1, 2, 5, 6, 11   
 */
vector<int> PED::realIndex(const vector<int> &complexIndex, const int N){
    vector<int> padComplexIndex = padEnds(complexIndex, -2, N);
    vector<int> a;
    a.reserve(N);
    for(auto it = padComplexIndex.begin(); it != padComplexIndex.end()-1; it++){ // note : -1 here.
	for(int i = *it+2; i < *(it+1); i++) {
	    a.push_back(i);
	}
    }
  
    return a;
}

/** @brief return the delta of a 2x2 matrix :
 *  for [a, b
 *       c, d]
 *   Delta = (a-d)^2 + 4bc
 */
double PED::deltaMat2(const Matrix2d &A){
    return (A(0,0) - A(1,1)) * (A(0,0) - A(1,1)) + 4 * A(1,0) * A(0,1);
}

/** @brief calculate the eigenvector of a 2x2 matrix which corresponds
 *   to the larger eigenvalue. 
 *  Note: MAKE SURE THAT THE INPUT MATRIX HAS REAL EIGENVALUES.
 */
Vector2d PED::vecMat2(const Matrix2d &A){
    EigenSolver<Matrix2d> eig(A); // eigenvalues must be real.
    Vector2d val = eig.eigenvalues().real();
    Matrix2d vec = eig.eigenvectors().real();
    Vector2d tmp;
  
    // chose the large eigenvalue in the upper place.
    if( val(0) > val(1) ) tmp = vec.col(0); 
    else tmp = vec.col(1);
  
    return tmp;
}

/** @brief get the eigenvalues and eigenvectors of 2x2 matrix
 *
 *  Eigenvalue is stored in exponential way : e^{mu + i*omega}.
 *  Here, omega is guarantted to be positive in [0, PI].
 *  The real and imaginary parts are splitted for eigenvector:
 *  [real(v), imag(v)]
 *  Only one eigenvalue and corresponding eigenvector are return.
 *  The other one is the conjugate.
 *  
 *  Example:
 *         for matrix  [1, -1
 *                      1,  1],
 *         it returns [ 0.346,  
 *                      0.785]
 *                and [-0.707,    0
 *                       0     0.707 ]
 *  Note : make sure that the input matrix has COMPLEX eigenvalues. 
 */
pair<Vector2d, Matrix2d> PED::complexEigsMat2(const Matrix2d &A){
    EigenSolver<Matrix2d> eig(A);
    Vector2cd val = eig.eigenvalues();
    Matrix2cd vec = eig.eigenvectors();

    Vector2d eigvals;
    Matrix2d eigvecs;
    eigvals(0) = log( abs(val(0)) );
    eigvals(1) = fabs( arg(val(0)) );
    int i = 0;
    if(arg(val(0)) < 0) i = 1;
    eigvecs.col(0) = vec.col(i).real();
    eigvecs.col(1) = vec.col(i).imag();

    return make_pair(eigvals, eigvecs);
}

/**
 * @brief obtain the 1x1 diagonal produce of a sequence of matrices
 *
 * @param[in] J       a sequence of matrices \f$ [J_m, ..., J_2, J_1] \f$
 * @param[in] k       position of the diagonal element
 * @return            [log product, sign]
 */
pair<double, int> PED::product1dDiag(const MatrixXd &J, const int k){
    const int N = J.rows();
    const int M = J.cols() / N;
    double logProduct = 0;
    int signProduct = 1;
    for(size_t i = 0; i < M; i++){
	logProduct += log( fabs(J(k, i*N+k)) );
	signProduct *= sgn(J(k, i*N+k));
    }
  
    return make_pair(logProduct, signProduct);
}

/**
 * @brief obtain the 2x2 diagonal produce of a sequence of matrices
 *
 * @param[in] J       a sequence of matrices \f$ [J_m, ..., J_2, J_1] \f$
 * @param[in] k       position of the diagonal element
 * @return            [log product, sign]
 */
pair<Matrix2d, double> PED::product2dDiag(const MatrixXd &J, const int k){
    const int N = J.rows();
    const int M = J.cols() / N;
    double logProduct = 0;
    Matrix2d A = MatrixXd::Identity(2,2);

    for(size_t i = 0; i < M; i++){
	A *= J.block<2,2>(k, i*N+k);
	double norm = A.cwiseAbs().maxCoeff();
	A = A.array() / norm;
	logProduct += log(norm);
    }
    return make_pair(A, logProduct);
}

/* @brief insert Householder transform between matrix product A*B -> A*H*H*B
 * The process also update the orthogonal matrix H : C -> C*H.
 * 
 * Here H is symmetric: H = I - 2vv* / (v*v). v = sign(x_1)||x||e_1 + x
 * A = A -2 (Av) v* / (v*v).
 * B = B - 2v (v*B) / (v*v)
 * Note : A denote the right cols of A, but B denotes the right bottom corner.
 * This process will change A, B and C.
 **/
void PED::HouseHolder(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, 
		      const int k, bool subDiag /* = false */){
    int shift = 0;
    if (subDiag) shift = 1;

    int br = A.rows() - k - shift; // block rows.
    int bc = A.cols() - k;  // block columns.
    VectorXd x = B.block(k + shift, k, br, 1); 
    int sx1 = sgn(x(0)); //sign of x(0)
    VectorXd e1 = VectorXd::Zero(br); e1(0) = 1;
    VectorXd v = sx1 * x.norm() * e1 + x;  
    double vnorm = v.norm(); v /= vnorm;

    A.rightCols(br) = A.rightCols(br) - 
	2 * (A.rightCols(br) * v) * v.transpose();

    C.rightCols(br) = C.rightCols(br) - 
	2 * (C.rightCols(br) * v) * v.transpose();

    B.bottomRightCorner(br, bc) = B.bottomRightCorner(br, bc) - 
	2 * v * (v.transpose() * B.bottomRightCorner(br, bc));
}

/** @brief return the sign of double precision number.
 */
int PED::sgn(const double &num){
    return (0 < num) - (num < 0);
}



/** @brief triDenseMat() creates the triplets of a dense matrix
 *  
 */
vector<Tri> PED::triDenseMat(const Ref<const MatrixXd> &A, const size_t M /* = 0 */, 
			     const size_t N /* = 0 */){
    size_t m = A.rows();
    size_t n = A.cols();

    vector<Tri> tri; 

    tri.reserve(m*n);
    for(size_t i = 0; i < n; i++) // cols
	for(size_t j = 0; j < m; j++) // rows
	    tri.push_back( Tri(M+j, N+i, A(j,i) ));

    return tri;
}

/** @brief triDenseMat() creates the triplets of the Kroneck product of
 *         an IxI identity matrix and a dense matrix: Identity x R
 *  
 */
vector<Tri> PED::triDenseMatKron(const size_t I, const Ref<const MatrixXd> &A, 
				 const size_t M /* = 0 */, const size_t N /* = 0 */){
    size_t m = A.rows();
    size_t n = A.cols();
  
    vector<Tri> nz; 
    nz.reserve(m*n*I);
  
    for(size_t i = 0; i < I; i++){
	vector<Tri> tri = triDenseMat(A, M+i*m, N+i*n);
	nz.insert(nz.end(), tri.begin(), tri.end());
    }
    return nz;
}

/** @brief triDiagMat() creates the triplets of a diagonal matrix.
 *
 */
vector<Tri> PED::triDiagMat(const size_t n, const double x, 
			    const size_t M /* = 0 */, const size_t N /* = 0 */ ){
    vector<Tri> tri;
    tri.reserve(n);
    for(size_t i = 0; i < n; i++) tri.push_back( Tri(M+i, N+i, x) );
    return tri;
}

/** @brief triDiagMat() creates the triplets of the product
 *         of a matrix and an IxI indentity matrix.
 *
 */
vector<Tri> PED::triDiagMatKron(const size_t I, const Ref<const MatrixXd> &A,
				const size_t M /* = 0 */, const size_t N /* = 0 */ ){
    size_t m = A.rows();
    size_t n = A.cols();
  
    vector<Tri> nz;
    nz.reserve(m*n*I);

    for(size_t i = 0; i < n; i++){
	for(size_t j = 0; j < m; j++){      
	    vector<Tri> tri = triDiagMat(I, A(j,i), M+j*I, N+i*I);
	    nz.insert(nz.end(), tri.begin(), tri.end());
	}
    }
  
    return nz;
}

/** @brief perSylvester() create the periodic Sylvester sparse matrix and the
 *         dense vector for the reordering algorithm.
 *
 *  @param J the sequence of matrix in order: Jm, Jm-1,... J1          
 *  @param P the position of the eigenvalue
 *  @param isReal for real eigenvector; otherwise, complex eigenvector
 */
pair<SpMat, VectorXd> PED::PerSylvester(const MatrixXd &J, const int &P, 
					const bool &isReal, const bool &Print){
    const int N = J.rows();
    const int M = J.cols() / N;
    if(isReal)
	{
	    // real case. only need to switch 1x1 matrix on the diagoanl
	    if(Print) printf("Forming periodic Sylvester matrix for a real eigenvalue:");
	    SpMat per_Sylvester(M*P, M*P);
	    VectorXd t12(M*P);
	    vector<Tri> nz; nz.reserve(2*M*P*P);
	    for(size_t i = 0; i < M; i++){
		if(Print) printf("%zd ", i);
		// Note: Ji is stored in the reverse way: Jm, Jm-1,...,J1
		t12.segment(i*P, P) = -J.block(0, (M-i-1)*N+P, P, 1); // vector -R^{12}
		vector<Tri> triR11 = triDenseMat( J.block(0, (M-i-1)*N, P, P), i*P, i*P );
		vector<Tri> triR22 = triDiagMat(P, -J(P, (M-i-1)*N+P), i*P, ((i+1)%M)*P);
		nz.insert(nz.end(), triR11.begin(), triR11.end());
		nz.insert(nz.end(), triR22.begin(), triR22.end());
	    }
	    if(Print) printf("\n");
	    per_Sylvester.setFromTriplets(nz.begin(), nz.end());
      
	    return make_pair(per_Sylvester, t12);
	}  
    else
	{
	    // complex case. Need to switch the 2x2 matrix on the diagonal.
	    if(Print) printf("Forming periodic Sylvester matrix for a complex eigenvalue:");
	    SpMat per_Sylvester(2*M*P, 2*M*P);
	    VectorXd t12(2*M*P);
	    vector<Tri> nz; nz.reserve(2*2*M*P*P);
	    for(size_t i = 0; i < M; i++){
		if(Print) printf("%zd ", i);
		// Note: Ji is stored in the reverse way: Jm, Jm-1,...,J1
		MatrixXd tmp = -J.block(0, (M-i-1)*N+P, P, 2); tmp.resize(2*P,1);
		t12.segment(i*2*P, 2*P) = tmp;
		vector<Tri> triR11 = triDenseMatKron(2, J.block(0, (M-i-1)*N, P, P), i*2*P, i*2*P);
		// Do not miss the transpose here.
		vector<Tri> triR22 = triDiagMatKron(P, -J.block(P, (M-i-1)*N+P, 2, 2).transpose(),
						    i*2*P, ((i+1)%M)*2*P);
		nz.insert(nz.end(), triR11.begin(), triR11.end());
		nz.insert(nz.end(), triR22.begin(), triR22.end());
	    }      
	    if(Print) printf("\n");
	    per_Sylvester.setFromTriplets(nz.begin(), nz.end());
      
	    return make_pair(per_Sylvester, t12);    
	}


}

/** @brief calculate eigenvector corresponding to the eigenvalue at 
 *         postion P given the Periodic Real Schur Form.
 *
 * The eigenvectors are not normalized, and they
 * correspond to matrix products:
 * JmJm-1..J1,  J1JmJm-1..J2, J2J1Jm...J3, ...
 *
 * Note, for complex eigenvectors. Their real and imaginary part are stored
 * separately.
 *
 * @param[in] J          (quasi-)upper triangular matrices [J_m,..., J_2, J_1]
 * @param[in] P          position of the diagonal element
 * @param[in] isReal     real or complex eignevector
 * @param[in] Print      Print info or not
 */
MatrixXd PED::oneEigVec(const MatrixXd &J, const int &P, 
			const bool &isReal, const bool &Print){
    const int N = J.rows();
    const int M = J.cols() / N;
    if(isReal)
	{
	    // create matrix to store the transfomr matrix

	    // First, put 1 at position P
	    MatrixXd ve = MatrixXd::Zero(N, M);
	    ve.row(P) = MatrixXd::Constant(1, M, 1.0);

	    // Second, put Px1 matrix X on position [0, P-1]
	    if(P != 0){
		pair<SpMat, VectorXd> tmp = PerSylvester(J, P, isReal, Print);
		SparseLU<SpMat, COLAMDOrdering<int> > lu(tmp.first);
		MatrixXd x = lu.solve(tmp.second); // in order to resize it, so set it matrix
		// not vector.
		x.resize(P, M);
		ve.topRows(P) = x; 
	    }
      
	    return ve;
	}
    else
	{
	    // create matrix to stored the transform matrix Sm,
	    // First, put 2x2 identity matrix at position [P,P+1].
	    MatrixXd Sm = MatrixXd::Zero(N, 2*M);
	    for(size_t i = 0; i < M; i++){
		Sm(P, i*2) = 1;
		Sm(P+1, i*2+1) = 1;
	    }
	    // second, put Px2 matrix X on position [0, P-1]
	    if(P != 0){
		pair<SpMat, VectorXd> tmp = PerSylvester(J, P, isReal, Print); 
		MatrixXd x = SparseLU<SpMat, COLAMDOrdering<int> >(tmp.first).solve(tmp.second);
		x.resize(P, 2*M);
		Sm.topRows(P) = x; 
	    }

	    //third, apply transform matrix Sm to the 2x2 eigenvectors. 
	    pair<Matrix2d, double> tmp = product2dDiag(J, P); 
	    pair<Vector2d, Matrix2d> tmp2 = complexEigsMat2(tmp.first);
	    Matrix2d v2 = tmp2.second; // real and imaginar parts of eigenvector 
	    for(size_t i = 0; i < M; i++){
		Sm.middleCols(2*i,2) *= v2; 
		//update the 2x2 eigenvector and normalize it.
		v2 = J.block<2,2>(P, (M-i-1)*N+P) * v2;
		// get the norm of a complex vector.
		double norm = sqrt(v2.col(0).squaredNorm() + v2.col(1).squaredNorm());
		v2 = v2.array() / norm;
	    }
      
	    return Sm;
	}
  
}

/** @brief fix the phase the complex eigenvectors
 *
 *  The complex eigenvectors got from periodic eigendecompostion 
 *  are not fixed becuase of the phase invariance. This method will 
 *  make the first element of each complex eigenvector be real.
 *
 *  @param[in,out] EigVecs Sequence of eigenvectors got from periodic
 *                 eigendecomposition
 *  @param[in] realCompelxIndex A vector indicating whether a vector is
 *             real or complex (third column of eigenvalues got fromx
 *             periodic eigendecomposition).
 */
void PED::fixPhase(MatrixXd &EigVecs, const VectorXd &realComplexIndex){
    const int N = sqrt(EigVecs.rows());
    const int M = EigVecs.cols();
    for(size_t i = 0; i < N; i++)
	{
	    if((int)realComplexIndex(i) == 1) {
		MatrixXcd cv = EigVecs.middleRows(i*N, N).array() * std::complex<double>(1,0) 
		    + std::complex<double>(0,1) * EigVecs.middleRows((i+1)*N, N).array();
		VectorXcd cj = cv.row(0).conjugate();
		for(size_t i = 0; i < M; i++) cj(i) /= abs(cj(i));
		cv = cv * cj.asDiagonal();
		EigVecs.middleRows(i*N, N) = cv.real();
		EigVecs.middleRows((i+1)*N, N) = cv.imag();
		i++;
	    }
      
	}
}

/** @brief Reverse the order of columns of a matrix.
 *
 *  J = [j1,j2,...jm] => [Jm, jm-1, ..., j1]
 *  @param[in,out] the matrix that need to be reversed in place.
 *  @see reverseOrderSize()
 */
void PED::reverseOrder(MatrixXd &J){
    const int rows = J.rows(), cols = J.cols(), M = cols / rows;
    MatrixXd tmp(rows, rows);
    for(int i = 0; i < M / 2; i++){
	tmp = J.middleCols(i*rows, rows);
	J.middleCols(i*rows, rows) = J.middleCols((M - i - 1)*rows, rows);
	J.middleCols((M - i - 1)*rows, rows) = tmp;
    }
}

/** @brief Reverse the order of a matrix and also resize it.
 *
 *  We usually squeeze the input such that each column represents
 *  one of the members of a sequence of matrices. reverseOrderSize()
 *  will first reverse the order and then resize it to the right format.
 *  Eg: [j1,j2,...jm] => [jm jm-1,...j1] with dimension [N^2, M]
 *  => dimension[N,N*M]
 *  @param[in,out] J The matrix that need to be reversed and
 *                   re-sized in place.
 *  @see reverseOrder()
 */
void PED::reverseOrderSize(MatrixXd &J){
    const int N = J.rows();
    const int M = J.cols();
    const int N2 = sqrt(N);
    reverseOrder(J);
    J.resize(N2, N2*M);
}

void PED::Trans(MatrixXd &J){
    const int N = J.rows();
    const int M = J.cols() / N;
    assert (M*N == J.cols());
  
    for(size_t i = 0; i < M/2; i++){
	MatrixXd tmp = J.middleCols(i*N, N);
	J.middleCols(i*N, N) = J.middleCols((M-i-1)*N, N).transpose();
	J.middleCols((M-i-1)*N, N) = tmp.transpose();
    }
  
    if(M % 2 == 1) { // deal with the middle matrix
	const int i = M / 2;
	MatrixXd tmp = J.middleCols(i*N, N);
	J.middleCols(i*N, N) = tmp.transpose();
    }
}

/**
 * @brief get ride of indices larger than trunc
 */
vector<int> 
PED::truncVec(const vector<int> &v, const int trunc){
    const int n = v.size();
    vector<int> v_trunc;
    v_trunc.reserve(n);
    for (int i = 0; i < n; i++) {
	if(v[i] <= trunc) v_trunc.push_back(v[i]);
    }
  
    return v_trunc;
}

/* ====================================================================== */
/*                      part related to power iteration                   */
/* ====================================================================== */

/**
 * @brief my wrapper to HouseHolderQR 
 */
std::pair<MatrixXd, MatrixXd>
PED::QR(const Ref<const MatrixXd> &A){
    int n = A.rows();
    int m = A.cols();
    assert(n >= m);

    // this is the only correct way to extract Q and R
    // which makes sure the returned matrices have continous memory layout 
    HouseholderQR<MatrixXd> qr(A);
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(n, m);
    MatrixXd R = MatrixXd::Identity(m, n) * qr.matrixQR().triangularView<Upper>();
    return std::make_pair(Q, R);
}


/**
 *  @param[in] J a sequence of matrices. Dimension [N, N*M].
 *
 *  This version of power iteration assumes the explicit form of
 *  original sequence.
 *  
 *  @see PowerIter0()
 */
std::tuple<MatrixXd, MatrixXd, MatrixXd, vector<int> >
PED::PowerIter(const Ref<const MatrixXd> &J, 
	       const Ref<const MatrixXd> &Q0,
	       const bool onlyLastQ,
	       int maxit, double Qtol, bool Print,
	       int PrintFreqency){
    const int N = J.rows();
    const int M = J.cols() / N; 
    const int M2 = Q0.cols();	
    
    // form the sequential QR funciton
    auto sqr = [&J, this, N, M, M2](MatrixXd Q, bool onlyLastQ) -> std::pair<MatrixXd, MatrixXd> {
	MatrixXd R(M2, M2*M);
	MatrixXd Qp;
	if(!onlyLastQ) Qp = MatrixXd(N, M2*M);
	for(int i = M-1; i >= 0; i--){ // start from the right side
	    auto qr = QR(J.middleCols(i*N, N) * Q);
	    if(!onlyLastQ) Qp.middleCols(i*M2, M2) = qr.first;
	    R.middleCols(i*M2, M2) = qr.second; 
	    Q = qr.first;
	}
	if(onlyLastQ) return std::make_pair(Q, R);
	else return std::make_pair(Qp, R);
    };
    
    // call the general funciton
    return PowerIter0(sqr, Q0, onlyLastQ, maxit, Qtol, Print, PrintFreqency);
}

/** 
 * @brief use power iteration to obtain the eigenvalues
 *
 * @see PowerEigE0()
 */
MatrixXd
PED::PowerEigE(const Ref<const MatrixXd> &J, 
	       const Ref<const MatrixXd> &Q0,
	       int maxit, double Qtol, bool Print,
	       int PrintFreqency){
    auto power = PowerIter(J, Q0, true, maxit, Qtol, Print, PrintFreqency);
    return getE(std::get<1>(power), std::get<3>(power));
}

/**
 * @brief check the convergence of the orthonormal matrices in the power iteration
 *
 * J*Q = Qp * R. Here we assume that each column of Q and Qp is normalized.
 * If Q converges, Qp = Q*D. Here D is a quasi-diagonal matrix.
 * D has diagonal element 1, -1, or 2x2 rotatio blocks whose determinant is 1 or -1.
 * 
 * Our criteria for convergence:
 *     real   =>     diagonal element is close to 1 and
 *                   subdiagonal elements are close to 0
 *     complext =>   determinant of 2x2 matrix is close to 1 and
 *                   subdiagonal elements are close to 0
 * Note, when the last column correpsonds to a complex one, we do not check it.
 *
 * @param[in] D     Q'*Qp
 * @param[in] Qtol  tolerance
 * @return          true => converged.  false => not converged
 */
bool
PED::checkQconverge(const Ref<const MatrixXd> &D, double Qtol){
    int N = D.rows();
    int M = D.cols();
    assert(N == M);
    
    for(size_t i = 0; i < M; i++ ){
	double er = fabs(D(i, i)) - 1;  
	if(fabs(er) < Qtol){	// real case
	    double er2 = (i < M -1) && (fabs(D(i+1, i)) > Qtol || fabs(D(i, i+1)) > Qtol);
	    if(er2) return false;
	}
	else{			//  complex pair case
	    if(i < M-1){	
		double ec = fabs(D(i, i)*D(i+1, i+1) - D(i+1, i)*D(i, i+1)) - 1;
		bool ec2 = (i < M -2) && (fabs(D(i+2, i+1)) > Qtol ||  fabs(D(i+1, i+2)) > Qtol);
		if(fabs(ec) > Qtol || ec2) return false;
		else i++;
	    }
	}
    }
    return true;
}

/**
 * @brief Get the complex eigenvalue positions for the power iteration
 *        method
 *
 * @note The last index will not be considered because it could not be
 *       if we want the full spectrum, or it should not be if only part of spectrum
 *       is wanted. Otherwise the next element will be retrieved and it
 *       gives memory error.
 * @see checkQconverge()
 */
std::vector<int>
PED::getCplPs(const Ref<const MatrixXd> D, double Qtol){
    int N = D.rows();
    int M = D.cols();
    assert(N == M);

    std::vector<int> ps;
    for(int i = 0; i < M-1; i++){ // note we use M-1
	double e = fabs(D(i, i)) - 1;  
	if(fabs(e) > Qtol) ps.push_back(i++);
    }

    return ps;
}
