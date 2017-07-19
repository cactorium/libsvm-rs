extern crate libsvm_sys;
extern crate libc;

use std::borrow::Borrow;

use std::ops::Deref;

use std::marker::PhantomData;

/// The node type used inside libsvm
pub type SvmNode = libsvm_sys::SvmNode;

/// A wrapper around the sparse vectors used in libsvm
pub struct SparseVector {
    pub inner: Vec<SvmNode>,
}

impl <'a> From<&'a [f64]> for SparseVector {
    fn from(fs: &[f64]) -> SparseVector {
        let vec = fs.iter()
            .enumerate()
            .filter(|&(_, x)| *x != 0.0)
            .map(|(idx, x)| SvmNode {
                index: idx as libc::c_int,
                value: *x as libc::c_double,
            })
            .chain(Some(SvmNode {
                index: -1 as libc::c_int,
                value: 0.0,
            }))
            .collect();
        SparseVector {
            inner: vec,
        }
    }
}

impl Deref for SparseVector {
    type Target = Vec<SvmNode>;
    fn deref(&self) -> &Vec<SvmNode> {
        &self.inner
    }
}

#[repr(C)]
pub enum SvmType {
    CSvc,
    NuSvc,
    OneClass,
    EpsilonSvr,
    NuSvr,
}

#[repr(C)]
pub enum KernelType {
    Linear,
    Poly,
    Rbf,
    Sigmoid,
    Precomputed,
}

impl SvmType {
    pub fn from_integral<T: Into<libc::c_int>>(t: T) -> Option<SvmType> {
        match t.into() {
            libsvm_sys::C_SVC =>       Some(SvmType::CSvc),
            libsvm_sys::NU_SVC =>      Some(SvmType::NuSvc),
            libsvm_sys::ONE_CLASS =>   Some(SvmType::OneClass),
            libsvm_sys::EPSILON_SVR => Some(SvmType::EpsilonSvr),
            libsvm_sys::NU_SVR =>      Some(SvmType::NuSvr),
            _ => None,
        }
    }
}

impl KernelType {
    pub fn from_integral<T: Into<libc::c_int>>(t: T) -> Option<KernelType> {
        match t.into() {
            libsvm_sys::LINEAR =>       Some(KernelType::Linear),
            libsvm_sys::POLY =>         Some(KernelType::Poly),
            libsvm_sys::RBF =>          Some(KernelType::Rbf),
            libsvm_sys::SIGMOID =>      Some(KernelType::Sigmoid),
            libsvm_sys::PRECOMPUTED =>  Some(KernelType::Precomputed),
            _ => None,
        }
    }
}


struct Problem<'a> {
    x: RefOrOwned<'a, SparseVector>,
    // needed to maintain the array of pointers in case svm_train uses that array
    x_raw: Vec<*const SvmNode>,
    y: RefOrOwned<'a, libc::c_double>,
}

enum RefOrOwned<'a, T: 'a> {
    Ref(&'a [T]),
    Owned(Vec<T>),
}

impl <'a, T> Deref for RefOrOwned<'a, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            &RefOrOwned::Ref(ref s) => s,
            &RefOrOwned::Owned(ref v) => &v,
        }
    }
}

pub struct Model<'a> {
    inner: *mut libsvm_sys::SvmModel,
    data: Option<Problem<'a>>,
}

pub struct Parameters<'a> {
    inner: libsvm_sys::SvmParameter,
    marker_: PhantomData<&'a [libc::c_double]>,
}

#[derive(Clone, Debug)]
pub enum ParameterCheckError {
    UnknownSvmType,
    UnknownKernelType,
    GammaNonpositive,
    KernelDegreeNonpositive,
    CacheSizeNonpositive,
    InvalidNu,
    PNonpositive,
    InvalidShrinking,
    InvalidProbability,
    Unsupported,
    InfeasibleNu,
}

#[derive(Clone, Debug)]
pub enum ModelCreationError {
    ParameterCheckError(ParameterCheckError),
    UnableToLoadModel,
    UnableToTrainModel,
}

#[derive(Clone, Debug)]
pub struct ModelSaveError;

impl <'a> Default for Parameters<'a> {
    fn default() -> Parameters<'a> {
        // TODO: use better default parameters
        Parameters {
            inner: libsvm_sys::SvmParameter {
                svm_type: SvmType::CSvc as libc::c_int,
                kernel_type: KernelType::Rbf as libc::c_int,
                degree: 0,
                gamma: 0.5,
                coef0: 0.0,

                cache_size: 64.0,
                eps: 1e-3,
                c: 1.0,
                nr_weight: 0,
                weight_label: 0 as *mut libc::c_int,
                weight: 0 as *mut libc::c_double,
                nu: 0.0,
                p: 0.0,
                shrinking: 0,
                probability: 0,
            },
            marker_: PhantomData,
        }
    }
}

/// A typesafe wrapper about libsvm that ensures that it is correctly
/// configured
impl <'a> Parameters<'a> {
    pub fn new() -> Parameters<'a> {
        Parameters::default()
    }

    pub fn linear(self) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.kernel_type = KernelType::Linear as libc::c_int;
        ret
    }

    pub fn poly(self, gamma: f64, coef0: f64, degree: isize) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.kernel_type = KernelType::Poly as libc::c_int;
        ret.inner.gamma = gamma as libc::c_double;
        ret.inner.coef0 = coef0 as libc::c_double;
        ret.inner.degree = degree as libc::c_int;
        ret
    }

    pub fn rbf(self, gamma: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.kernel_type = KernelType::Rbf as libc::c_int;
        ret.inner.gamma = gamma as libc::c_double;
        ret
    }

    pub fn sigmoid(self, gamma: f64, coef: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.kernel_type = KernelType::Sigmoid as libc::c_int;
        ret.inner.gamma = gamma as libc::c_double;
        ret.inner.coef0 = coef as libc::c_double;
        ret
    }

    /*
    pub fn precomputed(self) -> Parameters<'a> {
        // TODO: figure out a way to nicely wrap precomputed kernel semantics
        unimplemented!()
    }
    */

    pub fn c_svc<'b>(self,
                     c: f64,
                     idxs: &'b [libc::c_int],
                     weights: &'b [libc::c_double]) -> Parameters<'b> {
        let mut ret = Parameters {
            inner: self.inner,
            marker_: PhantomData,
        };
        ret.inner.svm_type = SvmType::CSvc as libc::c_int;
        ret.inner.c = c;
        assert_eq!(idxs.len(), weights.len());
        ret.inner.nr_weight = idxs.len() as libc::c_int;
        ret.inner.weight_label = idxs.as_ptr();
        ret.inner.weight = weights.as_ptr();
        ret
    }

    pub fn nu_svc(self, nu: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.svm_type = SvmType::NuSvc as libc::c_int;
        ret.inner.nu = nu as libc::c_double;
        ret
    }

    pub fn one_class(self, nu: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.svm_type = SvmType::OneClass as libc::c_int;
        ret.inner.nu = nu as libc::c_double;
        ret
    }

    pub fn epsilon_svr(self, c: f64, p: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.svm_type = SvmType::EpsilonSvr as libc::c_int;
        ret.inner.c = c as libc::c_double;
        ret.inner.p = p as libc::c_double;
        ret
    }

    pub fn nu_svr(self, c: f64, nu: f64) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.svm_type = SvmType::NuSvr as libc::c_int;
        ret.inner.c = c as libc::c_double;
        ret.inner.nu = nu as libc::c_double;
        ret
    }

    /// Sets the cache size used by libsvm in MB
    pub fn cache_size(self, sz: libc::c_double) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.cache_size = sz;
        ret
    }

    pub fn use_shrinking(self, b: bool) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.shrinking = if b { 1 } else { 0 };
        ret
    }

    pub fn estimate_probability(self, b: bool) -> Parameters<'a> {
        let mut ret = self;
        ret.inner.probability = if b { 1 } else { 0 };
        ret
    }
}

impl <'a> Model<'a> {
    pub fn train<'b, T, C>(data: &[(T, C)], parameters: &Parameters<'b>) -> Result<Model<'a>, ModelCreationError> where
        T: Borrow<[f64]>, C: Copy + Into<libc::c_double>
    {
        // copy the data into a vector of sparse vectors
        let sparse: Vec<SparseVector> = data.iter()
            .map(|&(ref x, _)| SparseVector::from(x.borrow()))
            .collect();
        let classes = data.iter()
            .map(|&(_, y)| y.into())
            .collect();
        let sparse_ptr = sparse.iter()
            .map(|x| x.inner.as_ptr())
            .collect();

        let problem = Problem {
            x: RefOrOwned::Owned(sparse),
            x_raw: sparse_ptr,
            y: RefOrOwned::Owned(classes),
        };
        let svm_problem = libsvm_sys::SvmProblem {
            l: problem.x.len() as libc::c_int,
            y: problem.y.as_ptr(),
            x: problem.x_raw.as_ptr(),
        };
        // check parameters
        let parameter_error = unsafe {
            libsvm_sys::svm_check_parameter(
                &svm_problem as *const libsvm_sys::SvmProblem,
                &parameters.inner as *const libsvm_sys::SvmParameter)
        };
        // check result
        if parameter_error != std::ptr::null() {
            // TODO
        }
        // train the SVM model
        let svm_model = unsafe {
            libsvm_sys::svm_train(
                &svm_problem as *const libsvm_sys::SvmProblem,
                &parameters.inner as *const libsvm_sys::SvmParameter)
        };
        // check result
        if svm_model.is_null() {
            Err(ModelCreationError::UnableToTrainModel)
        } else {
            Ok(Model {
                inner: svm_model,
                data: Some(problem),
            })
        }
    }

    /*
    pub fn train_sparse<'b, C>(data: &[(&'a SparseVector, C)], parameters: &Parameters<'b>) -> Result<Model<'a>, ModelCreationError> where
        C: Copy + Into<u32>
    {
        // TODO: allow for zero copy training
        unimplemented!()
    }
    */

    // FIXME: make more type safe; should be using a Path instead of a &str
    pub fn load_model(path: &str) -> Result<Model<'a>, ModelCreationError> {
        let model = unsafe {
            libsvm_sys::svm_load_model(path.as_ptr() as *const libc::c_char)
        };

        if model.is_null() {
            Err(ModelCreationError::UnableToLoadModel)
        } else {
            Ok(Model {
                inner: model,
                data: None,
            })
        }
    }

    pub fn save_model(&self, path: &str) -> Result<(), ModelSaveError> {
        let ret = unsafe {
            libsvm_sys::svm_save_model(
                path.as_ptr() as *const libc::c_char,
                self.inner as *const libsvm_sys::SvmModel)
        };

        if ret < 0 {
            Err(ModelSaveError)
        } else {
            Ok(())
        }
    }

    pub fn svm_type(&self) -> Option<SvmType> {
        let ret = unsafe {
            libsvm_sys::svm_get_svm_type(self.inner as *const libsvm_sys::SvmModel)
        };
        SvmType::from_integral(ret)
    }

    pub fn nr_class(&self) -> libc::c_int {
        unsafe {
            libsvm_sys::svm_get_nr_class(self.inner as *const libsvm_sys::SvmModel)
        }
    }

    // TODO: provide type safe wrappers for labels and sv_indices
    // need a way to see if the model has labels or sv_indices
    pub fn labels(&self) -> Vec<libc::c_int> {
        unimplemented!()
    }

    pub fn sv_indices(&self) -> Vec<libc::c_int> {
        unimplemented!()
    }

    pub fn predict(&self, data: &[f64]) -> libc::c_double {
        let sparse = SparseVector::from(data);
        unsafe {
            libsvm_sys::svm_predict(
                self.inner as *const libsvm_sys::SvmModel,
                sparse.as_ptr()
            )
        }
    }

    pub fn predict_sparse(&self, data: &SparseVector) -> libc::c_double {
        unsafe {
            libsvm_sys::svm_predict(
                self.inner as *const libsvm_sys::SvmModel,
                data.as_ptr()
            )
        }
    }
}

impl <'a> Drop for Model<'a> {
    fn drop(&mut self) {
        let mut ptr = self.inner as *mut libsvm_sys::SvmModel;
        unsafe {
            libsvm_sys::svm_free_and_destroy_model(&mut ptr as *mut *mut libsvm_sys::SvmModel);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let testx = vec![
            ([0.0, 0.0f64], -1),
            ([0.0, 1.0], -1),
            ([1.0, 0.0], 1),
            ([1.0, 1.0], 1),
        ];
        let parameters = Parameters::new()
            .c_svc(1.0, &[], &[]);
        let model = Model::train(&testx, &parameters).unwrap();
        let prediction = model.predict(&[0.0, -1.0]);
        assert_eq!(prediction, -1.0);
    }
}
