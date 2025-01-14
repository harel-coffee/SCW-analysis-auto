/* File: getcapemodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Tue Dec  1 05:34:12 2020
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *getcape_error;
static PyObject *getcape_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#define CHECKSCALAR(check,tcheck,name,show,var)\
    if (!(check)) {\
        char errstring[256];\
        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
        PyErr_SetString(getcape_error,errstring);\
        /*goto capi_fail;*/\
    } else 
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)
#else
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {
    PyObject* tmp = NULL;
    if (PyFloat_Check(obj)) {
#ifdef __sgi
        *v = PyFloat_AsDouble(obj);
#else
        *v = PyFloat_AS_DOUBLE(obj);
#endif
        return 1;
    }
    tmp = PyNumber_Float(obj);
    if (tmp) {
#ifdef __sgi
        *v = PyFloat_AsDouble(tmp);
#else
        *v = PyFloat_AS_DOUBLE(tmp);
#endif
        Py_DECREF(tmp);
        return 1;
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,"real");
    else if (PyString_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = getcape_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}

static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
    PyObject* tmp = NULL;
    if (PyInt_Check(obj)) {
        *v = (int)PyInt_AS_LONG(obj);
        return 1;
    }
    tmp = PyNumber_Int(obj);
    if (tmp) {
        *v = PyInt_AS_LONG(tmp);
        Py_DECREF(tmp);
        return 1;
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,"real");
    else if (PyString_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = getcape_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}

static int float_from_pyobj(float* v,PyObject *obj,const char *errmess) {
    double d=0.0;
    if (double_from_pyobj(&d,obj,errmess)) {
        *v = (float)d;
        return 1;
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(getcape,GETCAPE)(int*,float*,float*,float*,float*,int*,float*,int*,float*,float*);
extern void F_WRAPPEDFUNC(getqvs,GETQVS)(float*,float*,float*);
extern void F_WRAPPEDFUNC(getqvi,GETQVI)(float*,float*,float*);
extern void F_WRAPPEDFUNC(getthe,GETTHE)(float*,float*,float*,float*,float*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/********************************** getcape **********************************/
static char doc_f2py_rout_getcape_getcape[] = "\
cape,cin = getcape(p_in,t_in,td_in,pinc,source,ml_depth,adiabat,[nk])\n\nWrapper for ``getcape``.\
\n\nParameters\n----------\n"
"p_in : input rank-1 array('f') with bounds (nk)\n"
"t_in : input rank-1 array('f') with bounds (nk)\n"
"td_in : input rank-1 array('f') with bounds (nk)\n"
"pinc : input float\n"
"source : input int\n"
"ml_depth : input float\n"
"adiabat : input int\n"
"\nOther Parameters\n----------------\n"
"nk : input int, optional\n    Default: len(p_in)\n"
"\nReturns\n-------\n"
"cape : float\n"
"cin : float";
/* extern void F_FUNC(getcape,GETCAPE)(int*,float*,float*,float*,float*,int*,float*,int*,float*,float*); */
static PyObject *f2py_rout_getcape_getcape(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int*,float*,float*,float*,float*,int*,float*,int*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int nk = 0;
  PyObject *nk_capi = Py_None;
  float *p_in = NULL;
  npy_intp p_in_Dims[1] = {-1};
  const int p_in_Rank = 1;
  PyArrayObject *capi_p_in_tmp = NULL;
  int capi_p_in_intent = 0;
  PyObject *p_in_capi = Py_None;
  float *t_in = NULL;
  npy_intp t_in_Dims[1] = {-1};
  const int t_in_Rank = 1;
  PyArrayObject *capi_t_in_tmp = NULL;
  int capi_t_in_intent = 0;
  PyObject *t_in_capi = Py_None;
  float *td_in = NULL;
  npy_intp td_in_Dims[1] = {-1};
  const int td_in_Rank = 1;
  PyArrayObject *capi_td_in_tmp = NULL;
  int capi_td_in_intent = 0;
  PyObject *td_in_capi = Py_None;
  float pinc = 0;
  PyObject *pinc_capi = Py_None;
  int source = 0;
  PyObject *source_capi = Py_None;
  float ml_depth = 0;
  PyObject *ml_depth_capi = Py_None;
  int adiabat = 0;
  PyObject *adiabat_capi = Py_None;
  float cape = 0;
  float cin = 0;
  static char *capi_kwlist[] = {"p_in","t_in","td_in","pinc","source","ml_depth","adiabat","nk",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOOO|O:getcape.getcape",\
    capi_kwlist,&p_in_capi,&t_in_capi,&td_in_capi,&pinc_capi,&source_capi,&ml_depth_capi,&adiabat_capi,&nk_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable source */
    f2py_success = int_from_pyobj(&source,source_capi,"getcape.getcape() 5th argument (source) can't be converted to int");
  if (f2py_success) {
  /* Processing variable adiabat */
    f2py_success = int_from_pyobj(&adiabat,adiabat_capi,"getcape.getcape() 7th argument (adiabat) can't be converted to int");
  if (f2py_success) {
  /* Processing variable pinc */
    f2py_success = float_from_pyobj(&pinc,pinc_capi,"getcape.getcape() 4th argument (pinc) can't be converted to float");
  if (f2py_success) {
  /* Processing variable ml_depth */
    f2py_success = float_from_pyobj(&ml_depth,ml_depth_capi,"getcape.getcape() 6th argument (ml_depth) can't be converted to float");
  if (f2py_success) {
  /* Processing variable p_in */
  ;
  capi_p_in_intent |= F2PY_INTENT_IN;
  capi_p_in_tmp = array_from_pyobj(NPY_FLOAT,p_in_Dims,p_in_Rank,capi_p_in_intent,p_in_capi);
  if (capi_p_in_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(getcape_error,"failed in converting 1st argument `p_in' of getcape.getcape to C/Fortran array" );
  } else {
    p_in = (float *)(PyArray_DATA(capi_p_in_tmp));

  /* Processing variable cape */
  /* Processing variable cin */
  /* Processing variable nk */
  if (nk_capi == Py_None) nk = len(p_in); else
    f2py_success = int_from_pyobj(&nk,nk_capi,"getcape.getcape() 1st keyword (nk) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(len(p_in)>=nk,"len(p_in)>=nk","1st keyword nk","getcape:nk=%d",nk) {
  /* Processing variable t_in */
  t_in_Dims[0]=nk;
  capi_t_in_intent |= F2PY_INTENT_IN;
  capi_t_in_tmp = array_from_pyobj(NPY_FLOAT,t_in_Dims,t_in_Rank,capi_t_in_intent,t_in_capi);
  if (capi_t_in_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(getcape_error,"failed in converting 2nd argument `t_in' of getcape.getcape to C/Fortran array" );
  } else {
    t_in = (float *)(PyArray_DATA(capi_t_in_tmp));

  /* Processing variable td_in */
  td_in_Dims[0]=nk;
  capi_td_in_intent |= F2PY_INTENT_IN;
  capi_td_in_tmp = array_from_pyobj(NPY_FLOAT,td_in_Dims,td_in_Rank,capi_td_in_intent,td_in_capi);
  if (capi_td_in_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(getcape_error,"failed in converting 3rd argument `td_in' of getcape.getcape to C/Fortran array" );
  } else {
    td_in = (float *)(PyArray_DATA(capi_td_in_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&nk,p_in,t_in,td_in,&pinc,&source,&ml_depth,&adiabat,&cape,&cin);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("ff",cape,cin);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_td_in_tmp!=td_in_capi) {
    Py_XDECREF(capi_td_in_tmp); }
  }  /*if (capi_td_in_tmp == NULL) ... else of td_in*/
  /* End of cleaning variable td_in */
  if((PyObject *)capi_t_in_tmp!=t_in_capi) {
    Py_XDECREF(capi_t_in_tmp); }
  }  /*if (capi_t_in_tmp == NULL) ... else of t_in*/
  /* End of cleaning variable t_in */
  } /*CHECKSCALAR(len(p_in)>=nk)*/
  } /*if (f2py_success) of nk*/
  /* End of cleaning variable nk */
  /* End of cleaning variable cin */
  /* End of cleaning variable cape */
  if((PyObject *)capi_p_in_tmp!=p_in_capi) {
    Py_XDECREF(capi_p_in_tmp); }
  }  /*if (capi_p_in_tmp == NULL) ... else of p_in*/
  /* End of cleaning variable p_in */
  } /*if (f2py_success) of ml_depth*/
  /* End of cleaning variable ml_depth */
  } /*if (f2py_success) of pinc*/
  /* End of cleaning variable pinc */
  } /*if (f2py_success) of adiabat*/
  /* End of cleaning variable adiabat */
  } /*if (f2py_success) of source*/
  /* End of cleaning variable source */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of getcape *******************************/

/*********************************** getqvs ***********************************/
static char doc_f2py_rout_getcape_getqvs[] = "\
getqvs = getqvs(p,t)\n\nWrapper for ``getqvs``.\
\n\nParameters\n----------\n"
"p : input float\n"
"t : input float\n"
"\nReturns\n-------\n"
"getqvs : float";
/* extern void F_WRAPPEDFUNC(getqvs,GETQVS)(float*,float*,float*); */
static PyObject *f2py_rout_getcape_getqvs(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float getqvs = 0;
  float p = 0;
  PyObject *p_capi = Py_None;
  float t = 0;
  PyObject *t_capi = Py_None;
  static char *capi_kwlist[] = {"p","t",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO:getcape.getqvs",\
    capi_kwlist,&p_capi,&t_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable getqvs */
  /* Processing variable p */
    f2py_success = float_from_pyobj(&p,p_capi,"getcape.getqvs() 1st argument (p) can't be converted to float");
  if (f2py_success) {
  /* Processing variable t */
    f2py_success = float_from_pyobj(&t,t_capi,"getcape.getqvs() 2nd argument (t) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  (*f2py_func)(&getqvs,&p,&t);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("f",getqvs);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of t*/
  /* End of cleaning variable t */
  } /*if (f2py_success) of p*/
  /* End of cleaning variable p */
  /* End of cleaning variable getqvs */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of getqvs *******************************/

/*********************************** getqvi ***********************************/
static char doc_f2py_rout_getcape_getqvi[] = "\
getqvi = getqvi(p,t)\n\nWrapper for ``getqvi``.\
\n\nParameters\n----------\n"
"p : input float\n"
"t : input float\n"
"\nReturns\n-------\n"
"getqvi : float";
/* extern void F_WRAPPEDFUNC(getqvi,GETQVI)(float*,float*,float*); */
static PyObject *f2py_rout_getcape_getqvi(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float getqvi = 0;
  float p = 0;
  PyObject *p_capi = Py_None;
  float t = 0;
  PyObject *t_capi = Py_None;
  static char *capi_kwlist[] = {"p","t",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO:getcape.getqvi",\
    capi_kwlist,&p_capi,&t_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable getqvi */
  /* Processing variable p */
    f2py_success = float_from_pyobj(&p,p_capi,"getcape.getqvi() 1st argument (p) can't be converted to float");
  if (f2py_success) {
  /* Processing variable t */
    f2py_success = float_from_pyobj(&t,t_capi,"getcape.getqvi() 2nd argument (t) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  (*f2py_func)(&getqvi,&p,&t);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("f",getqvi);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of t*/
  /* End of cleaning variable t */
  } /*if (f2py_success) of p*/
  /* End of cleaning variable p */
  /* End of cleaning variable getqvi */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of getqvi *******************************/

/*********************************** getthe ***********************************/
static char doc_f2py_rout_getcape_getthe[] = "\
getthe = getthe(p,t,td,q)\n\nWrapper for ``getthe``.\
\n\nParameters\n----------\n"
"p : input float\n"
"t : input float\n"
"td : input float\n"
"q : input float\n"
"\nReturns\n-------\n"
"getthe : float";
/* extern void F_WRAPPEDFUNC(getthe,GETTHE)(float*,float*,float*,float*,float*); */
static PyObject *f2py_rout_getcape_getthe(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float getthe = 0;
  float p = 0;
  PyObject *p_capi = Py_None;
  float t = 0;
  PyObject *t_capi = Py_None;
  float td = 0;
  PyObject *td_capi = Py_None;
  float q = 0;
  PyObject *q_capi = Py_None;
  static char *capi_kwlist[] = {"p","t","td","q",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOO:getcape.getthe",\
    capi_kwlist,&p_capi,&t_capi,&td_capi,&q_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable getthe */
  /* Processing variable p */
    f2py_success = float_from_pyobj(&p,p_capi,"getcape.getthe() 1st argument (p) can't be converted to float");
  if (f2py_success) {
  /* Processing variable t */
    f2py_success = float_from_pyobj(&t,t_capi,"getcape.getthe() 2nd argument (t) can't be converted to float");
  if (f2py_success) {
  /* Processing variable td */
    f2py_success = float_from_pyobj(&td,td_capi,"getcape.getthe() 3rd argument (td) can't be converted to float");
  if (f2py_success) {
  /* Processing variable q */
    f2py_success = float_from_pyobj(&q,q_capi,"getcape.getthe() 4th argument (q) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  (*f2py_func)(&getthe,&p,&t,&td,&q);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("f",getthe);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of q*/
  /* End of cleaning variable q */
  } /*if (f2py_success) of td*/
  /* End of cleaning variable td */
  } /*if (f2py_success) of t*/
  /* End of cleaning variable t */
  } /*if (f2py_success) of p*/
  /* End of cleaning variable p */
  /* End of cleaning variable getthe */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of getthe *******************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"getcape",-1,{{-1}},0,(char *)F_FUNC(getcape,GETCAPE),(f2py_init_func)f2py_rout_getcape_getcape,doc_f2py_rout_getcape_getcape},
  {"getqvs",-1,{{-1}},0,(char *)F_WRAPPEDFUNC(getqvs,GETQVS),(f2py_init_func)f2py_rout_getcape_getqvs,doc_f2py_rout_getcape_getqvs},
  {"getqvi",-1,{{-1}},0,(char *)F_WRAPPEDFUNC(getqvi,GETQVI),(f2py_init_func)f2py_rout_getcape_getqvi,doc_f2py_rout_getcape_getqvi},
  {"getthe",-1,{{-1}},0,(char *)F_WRAPPEDFUNC(getthe,GETTHE),(f2py_init_func)f2py_rout_getcape_getthe,doc_f2py_rout_getcape_getthe},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "getcape",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyMODINIT_FUNC PyInit_getcape(void) {
#else
#define RETVAL
PyMODINIT_FUNC initgetcape(void) {
#endif
  int i;
  PyObject *m,*d, *s, *tmp;
#if PY_VERSION_HEX >= 0x03000000
  m = getcape_module = PyModule_Create(&moduledef);
#else
  m = getcape_module = Py_InitModule("getcape", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module getcape (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module 'getcape' is auto-generated with f2py (version:2).\nFunctions:\n"
"  cape,cin = getcape(p_in,t_in,td_in,pinc,source,ml_depth,adiabat,nk=len(p_in))\n"
"  getqvs = getqvs(p,t)\n"
"  getqvi = getqvi(p,t)\n"
"  getthe = getthe(p,t,td,q)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);
  getcape_error = PyErr_NewException ("getcape.error", NULL, NULL);
  /*
   * Store the error object inside the dict, so that it could get deallocated.
   * (in practice, this is a module, so it likely will not and cannot.)
   */
  PyDict_SetItemString(d, "_getcape_error", getcape_error);
  Py_DECREF(getcape_error);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
    tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
    PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
    Py_DECREF(tmp);
  }


    {
      extern float F_FUNC(getqvs,GETQVS)(void);
      PyObject* o = PyDict_GetItemString(d,"getqvs");
      tmp = F2PyCapsule_FromVoidPtr((void*)F_FUNC(getqvs,GETQVS),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
#if PY_VERSION_HEX >= 0x03000000
      s = PyUnicode_FromString("getqvs");
#else
      s = PyString_FromString("getqvs");
#endif
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    

    {
      extern float F_FUNC(getqvi,GETQVI)(void);
      PyObject* o = PyDict_GetItemString(d,"getqvi");
      tmp = F2PyCapsule_FromVoidPtr((void*)F_FUNC(getqvi,GETQVI),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
#if PY_VERSION_HEX >= 0x03000000
      s = PyUnicode_FromString("getqvi");
#else
      s = PyString_FromString("getqvi");
#endif
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    

    {
      extern float F_FUNC(getthe,GETTHE)(void);
      PyObject* o = PyDict_GetItemString(d,"getthe");
      tmp = F2PyCapsule_FromVoidPtr((void*)F_FUNC(getthe,GETTHE),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
#if PY_VERSION_HEX >= 0x03000000
      s = PyUnicode_FromString("getthe");
#else
      s = PyString_FromString("getthe");
#endif
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    
/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"getcape");
#endif
  return RETVAL;
}
#ifdef __cplusplus
}
#endif
