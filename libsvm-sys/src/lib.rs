extern crate libc;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SvmNode {
	pub index: libc::c_int,
	pub value: libc::c_double,
}

#[repr(C)]
pub struct SvmProblem {
	pub l: libc::c_int,
	pub y: *const libc::c_double,
	pub x: *const *const SvmNode,
}

/* svm_type */
pub const C_SVC: libc::c_int        = 0;
pub const NU_SVC: libc::c_int       = 1;
pub const ONE_CLASS: libc::c_int    = 2;
pub const EPSILON_SVR: libc::c_int  = 3;
pub const NU_SVR: libc::c_int       = 4;

/* kernel_type */
pub const LINEAR: libc::c_int       = 0;
pub const POLY: libc::c_int         = 1;
pub const RBF: libc::c_int          = 2;
pub const SIGMOID: libc::c_int      = 3;
pub const PRECOMPUTED: libc::c_int  = 4;

#[repr(C)]
pub struct SvmParameter {
	pub svm_type: libc::c_int,
	pub kernel_type: libc::c_int,
	pub degree: libc::c_int,	/* for poly */
	pub gamma: libc::c_double,	/* for poly/rbf/sigmoid */
	pub coef0: libc::c_double,	/* for poly/sigmoid */

	/* these are for training only */
	pub cache_size: libc::c_double, /* in MB */
	pub eps: libc::c_double,	/* stopping criteria */
	pub c: libc::c_double,	/* for C_SVC, EPSILON_SVR and NU_SVR */
	pub nr_weight: libc::c_int,		/* for C_SVC */
	pub weight_label: *const libc::c_int,	/* for C_SVC */
	pub weight: *const libc::c_double,		/* for C_SVC */
	pub nu: libc::c_double,	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	pub p: libc::c_double,	/* for EPSILON_SVR */
	pub shrinking: libc::c_int,         /* use the shrinking heuristics */
	pub probability: libc::c_int,       /* do probability estimates */
}

//
// svm_model
// 
#[repr(C)]
pub struct SvmModel {
	param: SvmParameter, 	/* parameter */
	nr_class: libc::c_int,		/* number of classes, = 2 in regression/one class svm */
	l: libc::c_int,			/* total #SV */
	sv: *mut *mut SvmNode,		/* SVs (SV[l]) */
	sv_coef: *mut *mut libc::c_double,	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	rho: *mut libc::c_double,		/* constants in decision functions (rho[k*(k-1)/2]) */
	prob_a: *mut libc::c_double,		/* pariwise probability information */
	prob_b: *mut libc::c_double,
	sv_indices: *mut libc::c_int,        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	label: *mut libc::c_int,		/* label of each class (label[k]) */
	n_sv: *mut libc::c_int,		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	free_sv: libc::c_int,		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
}

#[link(name = "svm")]
extern {

pub fn svm_train(prob: *const SvmProblem, param: *const SvmParameter) -> *mut SvmModel;
pub fn svm_cross_validation(prob: *const SvmProblem, param: *const SvmParameter, nr_fold: libc::c_int, target: *mut libc::c_double);

pub fn svm_save_model(model_file_name: *const libc::c_char, model: *const SvmModel) -> libc::c_int;
pub fn svm_load_model(model_file_name: *const libc::c_char) -> *mut SvmModel;

pub fn svm_get_svm_type(model: *const SvmModel) -> libc::c_int;
pub fn svm_get_nr_class(model: *const SvmModel) -> libc::c_int;
pub fn svm_get_labels(model: *const SvmModel, label: *mut libc::c_int);
pub fn svm_get_sv_indices(model: *const SvmModel, sv_indices: *mut libc::c_int);
pub fn svm_get_nr_sv(model: *const SvmModel) -> libc::c_int;
pub fn svm_get_svr_probability(model: *const SvmModel) -> libc::c_double;

pub fn svm_predict_values(model: *const SvmModel, x: *const SvmNode, dec_values: *mut libc::c_double) -> libc::c_double;
pub fn svm_predict(model: *const SvmModel, x: *const SvmNode) -> libc::c_double;
pub fn svm_predict_probability(model: *const SvmModel, x: *const SvmNode, prob_estimates: *mut libc::c_double) -> libc::c_double;

pub fn svm_free_model_content(model_ptr: *mut SvmModel);
pub fn svm_free_and_destroy_model(model_ptr_ptr: *mut *mut SvmModel);
pub fn svm_destroy_param(param: *mut SvmParameter);

pub fn svm_check_parameter(prob: *const SvmProblem, param: *const SvmParameter) -> *const libc::c_char;
pub fn svm_check_probability_model(model: *const SvmModel) -> libc::c_int;

pub fn svm_set_print_string_function(print_fn: extern fn (*const libc::c_char));

}
