#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Dense>
#include <cstdio>

#include "ped.hpp"

using namespace std;
using namespace Eigen;
namespace bp = boost::python;
namespace bn = boost::numpy;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/* get the dimension of an array */
inline void getDims(bn::ndarray x, int &m, int &n){
    if(x.get_nd() == 1){
	m = 1;
	n = x.shape(0);
    } else {
	m = x.shape(0);
	n = x.shape(1);
    }
}

/*
 * @brief used to copy content in Eigen array to boost.numpy array.
 *
 *  Only work for double array/matrix
 */
inline bn::ndarray copy2bn(const Ref<const ArrayXXd> &x){
    int m = x.cols();
    int n = x.rows();

    Py_intptr_t dims[2];
    int ndim;
    if(m == 1){
	ndim = 1;
	dims[0] = n;
    }
    else {
	ndim = 2;
	dims[0] = m;
	dims[1] = n;
    }
    bn::ndarray px = bn::empty(ndim, dims, bn::dtype::get_builtin<double>());
    memcpy((void*)px.get_data(), (void*)x.data(), sizeof(double) * m * n);
	    
    return px;
}

/**
 * @brief std::vector to bp::list
 */
template <class T>
bp::list toList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
	list.append(*iter);
    }
    return list;
}

template <class T>
std::vector<T> toVector(bp::list bl){
    std::vector<T> v;
    for(int i = 0; i < bp::len(bl); i++){
	v.push_back( bp::extract<T>(bl[i]) );
    }
    return v;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class pyPED : public PED {
  
public:

    /* PowerIter */
    bp::tuple PYPowerIter(bn::ndarray J, bn::ndarray Q, 
			  bool onlyLastQ,
			  int maxit, double Qtol, bool Print,
			  int PrintFreqency){
	int m, n;
	getDims(J, m, n);
	Map<MatrixXd> tmpJ((double*)J.get_data(), n, m);
	getDims(Q, m, n);
	Map<MatrixXd> tmpQ((double*)Q.get_data(), n, m); 
	
	auto tmp = PowerIter(tmpJ, tmpQ, onlyLastQ, maxit, Qtol, Print, PrintFreqency);
	return bp::make_tuple(copy2bn(std::get<0>(tmp)), 
			      copy2bn(std::get<1>(tmp)),
			      copy2bn(std::get<2>(tmp)),
			      toList(std::get<3>(tmp))   
			      );
    }

    /* getE */
    bn::ndarray PYgetE(bn::ndarray R, bp::list ci){
	int m, n;
	getDims(R, m, n);
	Map<MatrixXd> tmpR((double*)R.get_data(), n, m);
	return copy2bn(getE(tmpR, toVector<int>(ci)));
    }

    /* QR */
    bp::tuple PYQR(bn::ndarray A){
	int m, n;
	getDims(A, m, n);
	Map<MatrixXd> tmpA((double*)A.get_data(), n, m);
	auto tmp = QR(tmpA);
	return bp::make_tuple(copy2bn(tmp.first),
			      copy2bn(tmp.second)
			      );
    }

    /* PowerEigE */
    bn::ndarray PYPowerEigE(bn::ndarray J, bn::ndarray Q, 
			    int maxit, double Qtol, bool Print,
			    int PrintFreqency){
	int m, n;
	getDims(J, m, n);
	Map<MatrixXd> tmpJ((double*)J.get_data(), n, m);
	getDims(Q, m, n);
	Map<MatrixXd> tmpQ((double*)Q.get_data(), n, m); 
	
	auto tmp = PowerEigE(tmpJ, tmpQ, maxit, Qtol, Print, PrintFreqency);
	return copy2bn(tmp);	
    }

};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(py_ped) {
    bn::initialize();

    // must provide the constructor
    bp::class_<PED>("PED")
	;
    
    bp::class_<pyPED, bp::bases<PED> >("pyPED")
	.def("PowerIter", &pyPED::PYPowerIter)
	.def("QR", &pyPED::PYQR)
	.def("PowerEigE", &pyPED::PYPowerEigE)
	.def("getE", &pyPED::PYgetE)
	;

}
