/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 //
// Template math library for common 3D functionality
//
// nvMatrix.h - template matrix code
//
// This code is in part deriver from glh, a cross platform glut helper library.
// The copyright for glh follows this notice.
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

/*
    Copyright (c) 2000 Cass Everitt
	Copyright (c) 2000 NVIDIA Corporation
    All rights reserved.

    Redistribution and use in source and binary forms, with or
	without modification, are permitted provided that the following
	conditions are met:

     * Redistributions of source code must retain the above
	   copyright notice, this list of conditions and the following
	   disclaimer.

     * Redistributions in binary form must reproduce the above
	   copyright notice, this list of conditions and the following
	   disclaimer in the documentation and/or other materials
	   provided with the distribution.

     * The names of contributors to this software may not be used
	   to endorse or promote products derived from this software
	   without specific prior written permission. 

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
	   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
	   REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
	   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
	   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
	   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
	   POSSIBILITY OF SUCH DAMAGE. 


    Cass Everitt - cass@r3.nu
*/

#ifndef NV_MATRIX_H
#define NV_MATRIX_H

#define _USE_MATH_DEFINES
#include <math.h>

//using namespace std;
namespace nv {

template <class T> class vec2;
template <class T> class vec3;
template <class T> class vec4;

////////////////////////////////////////////////////////////////////////////////
//
//  Matrix
//
////////////////////////////////////////////////////////////////////////////////
template<class T>
class matrix4
{

public:

    matrix4() { setIdentity(); }

    matrix4( T t ) 
    { set(t); }

    matrix4( const T * m )
    { set_value(m); }

    matrix4( T a00, T a01, T a02, T a03,
        T a10, T a11, T a12, T a13,
        T a20, T a21, T a22, T a23,
        T a30, T a31, T a32, T a33 ) :
    m00(a00), m01(a01), m02(a02), m03(a03),
    m10(a10), m11(a11), m12(a12), m13(a13),
    m20(a20), m21(a21), m22(a22), m23(a23),
    m30(a30), m31(a31), m32(a32), m33(a33)
    {}


    void get( T * mp ) const {
        int c = 0;
        for(int j=0; j < 4; j++)
            for(int i=0; i < 4; i++)
                mp[c++] = element(i,j);
    }

    const T * getPtr() const {
        return _array;
    }

    void set( T * mp) {
        int c = 0;
        for(int j=0; j < 4; j++)
            for(int i=0; i < 4; i++)
                element(i,j) = mp[c++];
    }

    void set( T r ) {
        for(int i=0; i < 4; i++)
            for(int j=0; j < 4; j++)
                element(i,j) = r;
    }

    void setIdentity() {
        element(0,0) = 1.0;
        element(0,1) = 0.0;
        element(0,2) = 0.0; 
        element(0,3) = 0.0;

        element(1,0) = 0.0;
        element(1,1) = 1.0; 
        element(1,2) = 0.0;
        element(1,3) = 0.0;

        element(2,0) = 0.0;
        element(2,1) = 0.0;
        element(2,2) = 1.0;
        element(2,3) = 0.0;

        element(3,0) = 0.0; 
        element(3,1) = 0.0; 
        element(3,2) = 0.0;
        element(3,3) = 1.0;
    }

    // set a uniform scale
    void scale( T s ) {
        element(0,0) = s;
        element(1,1) = s;
        element(2,2) = s;
    }

    void scale( const vec3<T> & s ) {
        for (int i = 0; i < 3; i++) element(i,i) = s[i];
    }


    void translate( const vec3<T> & t ) {
        for (int i = 0; i < 3; i++) element(i,3) = t[i];
    }

	void setRow(int r, const vec3<T> & t) {
        for (int i = 0; i < 3; i++) element(r,i) = t[i];
			
		element(r,4) = 0.0f;
    }
	
    void setCol(int c, const vec3<T> & t) {
        for (int i = 0; i < 3; i++) element(i,c) = t[i];
		
		element(4,c) = 0.0f;
    }
	
    void setRow(int r, const vec4<T> & t) {
        for (int i = 0; i < 4; i++) element(r,i) = t[i];
    }

    void setCol(int c, const vec4<T> & t) {
        for (int i = 0; i < 4; i++) element(i,c) = t[i];
    }

    vec4<T> getRow(int r) const {
        vec4<T> v;
        for (int i = 0; i < 4; i++) v[i] = element(r,i);
        return v;
    }

    vec4<T> getCol(int c) const {
        vec4<T> v;
        for (int i = 0; i < 4; i++) v[i] = element(i,c);
        return v;
    }

    friend matrix4 inverted( const matrix4<T> & m) {
        matrix4<T> minv;

        T r1[8], r2[8], r3[8], r4[8];
        T* s[4], *tmprow;

        s[0] = &r1[0];
        s[1] = &r2[0];
        s[2] = &r3[0];
        s[3] = &r4[0];

        register int i,j,p,jj;
        for(i=0;i<4;i++) {
            for(j=0;j<4;j++) {
                s[i][j] = m.element(i,j);
                if(i==j) s[i][j+4] = 1.0;
                else     s[i][j+4] = 0.0;
            }
        }
        T scp[4];
        for(i=0;i<4;i++) {
            scp[i] = T(fabs(s[i][0]));
            for(j=1;j<4;j++)
                if(T(fabs(s[i][j])) > scp[i]) scp[i] = T(fabs(s[i][j]));
            if(scp[i] == 0.0) return minv; // singular matrix!
        }

        int pivot_to;
        T scp_max;
        for(i=0;i<4;i++) {
            // select pivot row
            pivot_to = i;
            scp_max = T(fabs(s[i][i]/scp[i]));
            // find out which row should be on top
            for(p=i+1;p<4;p++)
                if (T(fabs(s[p][i]/scp[p])) > scp_max) {
                    scp_max = T(fabs(s[p][i]/scp[p]));
                    pivot_to = p;
                }
            // Pivot if necessary
            if(pivot_to != i) {
                tmprow = s[i];
                s[i] = s[pivot_to];
                s[pivot_to] = tmprow;
                T tmpscp;
                tmpscp = scp[i];
                scp[i] = scp[pivot_to];
                scp[pivot_to] = tmpscp;
            }

            T mji;
            // perform gaussian elimination
            for(j=i+1;j<4;j++) {
                mji = s[j][i]/s[i][i];
                s[j][i] = 0.0;
                for(jj=i+1;jj<8;jj++)
                    s[j][jj] -= mji*s[i][jj];
            }
        }
        if(s[3][3] == 0.0) return minv; // singular matrix!

        //
        // Now we have an upper triangular matrix.
        //
        //  x x x x | y y y y
        //  0 x x x | y y y y 
        //  0 0 x x | y y y y
        //  0 0 0 x | y y y y
        //
        //  we'll back substitute to get the inverse
        //
        //  1 0 0 0 | z z z z
        //  0 1 0 0 | z z z z
        //  0 0 1 0 | z z z z
        //  0 0 0 1 | z z z z 
        //

        T mij;
        for(i=3;i>0;i--) {
            for(j=i-1;j > -1; j--) {
                mij = s[j][i]/s[i][i];
                for(jj=j+1;jj<8;jj++)
                    s[j][jj] -= mij*s[i][jj];
            }
        }

        for(i=0;i<4;i++)
            for(j=0;j<4;j++)
                minv(i,j) = s[i][j+4] / s[i][i];

        return minv;
    }


    friend matrix4 transposed( const matrix4<T> & m) {
        matrix4<T> mtrans;

        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                mtrans(i,j) = m.element(j,i);		
        return mtrans;
    }


	friend matrix4<T> rotation(const vec3<T> ax3, T angle){
		T rad = angle/180*(float)M_PI;
		T c = cos(rad);
		T s = sin(rad);
		T x = ax3.x;
		T y = ax3.y;
		T z = ax3.z;
		
		matrix4<T> mat;
		mat.m00 = x*x*(1-c)+c;		mat.m01 = y*x*(1-c)-z*s;	mat.m02 = x*z*(1-c)+y*s;	mat.m03 = 0;
		mat.m10 = y*x*(1-c)+z*s;	mat.m11 = y*y*(1-c)+c;		mat.m12 = y*z*(1-c)-x*s;	mat.m13 = 0;
		mat.m20 = z*x*(1-c)-y*s;	mat.m21 = z*y*(1-c)+x*s;	mat.m22 = z*z*(1-c)+c;		mat.m23 = 0;
		mat.m30 = 0;				mat.m31 = 0;				mat.m32 = 0;				mat.m33 = 1;
		
		return mat;
	}
	
    matrix4<T> & operator *= ( const matrix4<T> & rhs ) {
        matrix4<T> mt(*this);
        set_value(T(0));

        for(int i=0; i < 4; i++)
            for(int j=0; j < 4; j++)
                for(int c=0; c < 4; c++)
                    element(i,j) += mt(i,c) * rhs(c,j);
        return *this;
    }

    friend matrix4<T> operator * ( const matrix4<T> & lhs, const matrix4<T> & rhs ) {
        matrix4<T> r(T(0));

        for(int i=0; i < 4; i++)
            for(int j=0; j < 4; j++)
                for(int c=0; c < 4; c++)
                    r.element(i,j) += lhs(i,c) * rhs(c,j);
        return r;
    }


    vec3<T> operator *( const vec3<T> &src) const {
        vec3<T> r;
        for ( int i = 0; i < 3; i++)
            r[i]  = ( src[0] * element(i,0) + src[1] * element(i,1) +
					 src[2] * element(i,2));
        return r;
    }
	
	vec4<T> operator *( const vec4<T> &src) const {
        vec4<T> r;
        for ( int i = 0; i < 4; i++)
            r[i]  = ( src[0] * element(i,0) + src[1] * element(i,1) +
                      src[2] * element(i,2) + src[3] * element(i,3));
        return r;
    }

    friend vec4<T> operator *( const vec4<T> &lhs, const matrix4 &rhs) {
        vec4<T> r;
        for ( int i = 0; i < 4; i++)
            r[i]  = ( lhs[0] * rhs.element(0,i) + lhs[1] * rhs.element(1,i) +
                      lhs[2] * rhs.element(2,i) + lhs[3] * rhs.element(3,i));
        return r;
    }

    T & operator () (int row, int col) {
        return element(row,col);
    }

    const T & operator () (int row, int col) const {
        return element(row,col);
    }

    T & element (int row, int col) {
        return _array[row | (col<<2)];
    }

    const T & element (int row, int col) const {
        return _array[row | (col<<2)];
    }

    matrix4 & operator *= ( const T & r ) {
        for (int i = 0; i < 4; ++i) {
            element(0,i) *= r;
            element(1,i) *= r;
            element(2,i) *= r;
            element(3,i) *= r;
        }
        return *this;
    }

    matrix4 & operator += ( const matrix4 & mat ) {
        for (int i = 0; i < 4; ++i) {
            element(0,i) += mat.element(0,i);
            element(1,i) += mat.element(1,i);
            element(2,i) += mat.element(2,i);
            element(3,i) += mat.element(3,i);
        }
        return *this;
    }

    
    friend bool operator == ( const matrix4 & lhs, const matrix4 & rhs ) {
        bool r = true;
        for (int i = 0; i < 16; i++)
            r &= lhs._array[i] == rhs._array[i];
        return r;
    }

    friend bool operator != ( const matrix4 & lhs, const matrix4 & rhs )  {
        bool r = true;
        for (int i = 0; i < 16; i++)
            r &= lhs._array[i] != rhs._array[i];
        return r;
    }

    union {
        struct {
            T m00, m01, m02, m03;   // standard names for components
            T m10, m11, m12, m13;   // standard names for components
            T m20, m21, m22, m23;   // standard names for components
            T m30, m31, m32, m33;   // standard names for components
        };
        T _array[16];     // array access
    };
};

};

#endif
