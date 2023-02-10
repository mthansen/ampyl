#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:45:36 2023

@author: s1889602
"""

import numpy as np
import unittest
import ampyl
from scipy.linalg import block_diag
from copy import deepcopy

ell = 1.0 
Group = ampyl.Groups(ell) 
OhP = Group.OhP
NVEC_CUTOFF = 1
TWOPI = 2*np.pi
LIMIT = 10e-13

C_MAT = np.array([[1./np.sqrt(10), 1./np.sqrt(10), 1./np.sqrt(10), 1./np.sqrt(10), 1./np.sqrt(10), 1./np.sqrt(10), np.sqrt(2/5)],\
                 [-1./2., 0, -1./2., 1./2., 0., 1./2., 0. ],\
                 [-1./np.sqrt(12), -1./np.sqrt(3), 1./np.sqrt(12), -1./np.sqrt(12), 1./np.sqrt(3), 1./np.sqrt(12), 0.],\
                [np.sqrt(3/5)/2, -1./np.sqrt(15), np.sqrt(3/5)/2, np.sqrt(3/5)/2, -1/np.sqrt(15), np.sqrt(3/5)/2, -2/np.sqrt(15) ],\
                [1./2., 0, -1./2., -1./2., 0., 1./2., 0.],\
                [0., 1/np.sqrt(3), 0., 0., 1/np.sqrt(3), 0, -1/np.sqrt(3)],\
                [-1/np.sqrt(6), 1/np.sqrt(6), 1/np.sqrt(6),  -1/np.sqrt(6), -1/np.sqrt(6), 1/np.sqrt(6), 0.]])




class Nonint_Proj:
    
    def __init__(self, Emax= 5.0 , nP=np.array([0,0,0]), Lmax = 7.0, masses = 3*[1]):
        self.Emax = Emax
        self.nP = nP
        self.Lmax = Lmax
        self.masses = masses
        

    def get_nvec_cutoff_n3zero(self, Emax, nP, Lmax, masses):
        pSQ = 0.0
        if len(masses) == 3:
            m = masses[-1]
            pSQ = ((Emax - m)**2 - (nP@nP)*(TWOPI/Lmax)**2)/4 - m**2
        if len(masses) == 2:
            m = masses[1] 
            pSQ = Emax**2/4 - m**2
        nSQ = pSQ*(Lmax/TWOPI)**2
        return int(np.sqrt(nSQ))

    def nvecs_generator(self, nvec_cutoff):
        """
        
        Generates all possible combonation of 3-integer vector 
        that are below the NVEC_CUTOFF
        
        input:
        NVEC_CUTOFF[int]      -- the maximum NVEC_CUTOFF for the component of the 3-vector of the particles
        
        output[nd.array] -- a nd array of size ((2(NVEC_CUTOFF)+1)^3 , 3) 
        
        
        """
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng] * 3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return nvecs


    def get_E_nSQ_vecs(self, Emax, nP, Lmax, nvec_cutoff, m1,m2,m3, identical=False):
        """
        
        Get the non-interacting energies and the square of the 3-integer-vectors of the particles. 
        Also, get the multiplicites of the energies 
        
        input:
        Emax[float]       -- the maximum energy of the system
        nP[np.array]      -- the 3-integer vector for the total momentum 
        Lmax[float]       -- the box length
        nNVEC_CUTOFF[int]  -- the maximum NVEC_CUTOFF for the component of the 3-vector of the particles
        m1[float]         -- mass of the 1st particle
        m2[float]         -- mass of the 2nd particle
        m3[float]         -- mass of the 3rd particle
        
        output:
        E_nSQ_unique[nd.array]  -- 4-vector [E, n1SQ, n2SQ, n3SQ] of size (?,4) sorted from lowest to highest energy
        counts_sorted[np.array] -- the multiplicites of the non-interacting energies
        
        """
        nvecs = self.nvecs_generator(nvec_cutoff)
        E_nSQ_arr = []
        n_arr = []
        for n1 in nvecs:
            for n2 in nvecs:
                n3 = nP - n1 - n2
                n1SQ = (n1@n1)
                n2SQ = (n2@n2)
                n3SQ = (n3@n3)
                TWOPI_L_SQ = (TWOPI/Lmax)**2 
                E = np.sqrt(m1**2 + TWOPI_L_SQ*n1SQ) + \
                    np.sqrt(m2**2 + TWOPI_L_SQ*n2SQ) + \
                    np.sqrt(m3**2 + TWOPI_L_SQ*n3SQ)
                if E <= Emax:
                    nSQ_list = list(np.sort([n1SQ, n2SQ, n3SQ]))
                    E_nSQ_arr = E_nSQ_arr + [[E, *nSQ_list]]
                    n_arr = n_arr + [[n1,n2,n3]]  
    
        n_arr = np.array(n_arr)
        E_nSQ_arr = np.array(E_nSQ_arr)
    
        E_unique, E_index, E_counts = np.unique(E_nSQ_arr[:,0], return_index=True, return_counts = True)
        E_nSQ_unique = E_nSQ_arr[E_index]
    
        nSQ_arr_unique = E_nSQ_unique[:,[1,2,3]]
        n_arr_sorted = len(nSQ_arr_unique)*[[]] #Find a better way of sorting them
    
        if identical:
            n_arr_identical = np.unique(np.sort(n_arr, axis=1), axis = 0)
            for i in range(len(nSQ_arr_unique)):
                n_sorted_tmp = []
                for n in n_arr_identical:
                    nSQ_tmp = np.sort(np.sum(n**2, axis=1), axis=0)
                    if (np.sort(nSQ_arr_unique[i]) == nSQ_tmp).all():
                            n_sorted_tmp.append(n)
                n_arr_sorted[i] = np.array(n_sorted_tmp) 
    
        if not identical:
            for i in range(len(nSQ_arr_unique)):
                n_sorted_tmp = []
                for n in n_arr:
                    nSQ_tmp = np.sort(np.sum(n**2, axis=1), axis=0)
                    if (np.sort(nSQ_arr_unique[i]) == nSQ_tmp).all():
                        n_sorted_tmp.append(n)
    
                n_arr_sorted[i] = np.array(n_sorted_tmp)     
    
    
        n_arr_sorted = np.array(n_arr_sorted, dtype=object)
        
        return E_nSQ_unique, E_counts, n_arr_sorted


    def get_induced_rep(self, n_arr, g, identical = False):
        '''
        
    
        Parameters
        ----------
        n_arr : np.ndarray
            A list of three 3-integer vector and all its possible rotation by the governing symmetry.
        g : np.ndarray
            A 3x3 rotation matrix from the symmetry group.
        identical : bool, optional
            DESCRIPTION. The default is False for non-identical particles 
            and True in the case of identical particles.
    
        Returns
        -------
        ind_rep : np.ndarray
            An induced representation of the n_arr. It is a square matrix of size len(n_arr)xlen(n_arr).
    
        '''
        ind_rep = np.zeros((len(n_arr), len(n_arr)))
        if identical:
            for k in range(len(n_arr)):
                n_rot = (g@n_arr[k].T).T
                n_rot_sorted = np.sort(n_rot, axis=0)
                if (n_rot_sorted == n_arr[k]).all():
                    ind_rep[k][k] = 1.0
                else:
                    for i in range(len(n_arr)):
                        if (n_arr[i] == n_rot_sorted).all():
                            ind_rep[i][k] = 1.0
        if not identical:
            for k in range(len(n_arr)):
                n_rot = (g@n_arr[k].T).T
                if (n_rot == n_arr[k]).all():
                    ind_rep[k][k] = 1.0
                else:
                    for i in range(len(n_arr)):
                        if (n_arr[i] == n_rot).all():
                            ind_rep[i][k] = 1.0
        return ind_rep

    def sort_by_rot(self, nvec_arr):
        dim = len(nvec_arr[0])
        little_group = dim*[[]]
    
        for g in nvec_arr:
            rot_arr = np.array(dim*[1])@np.absolute(g)
            ind_loc = np.where(rot_arr != 0.0)
            little_group[ind_loc[0][0]] = little_group[ind_loc[0][0]] + [g]
        little_group = np.array(little_group, dtype=object)
    
        lg = []
        for i in range(len(little_group)):
            for elem in little_group[i]:
                lg.append(elem)
        return np.array(lg)

    def get_total_proj(self, Emax, nP, Lmax, nvec_cutoff , masses, g, state):
        '''
        
    
        Parameters
        ----------
        Emax : float
            Maximum NVEC_CUTOFF energy.
        nP : np.array
            the total momentum of the volume in units of (2*pi/L).
        Lmax : float
            Maximum NVEC_CUTOFF volume extent.
        nvec_NVEC_CUTOFF : int
            A cuttoff for the squared magnitude of the particle momenutm.
        m1 : float
            The mass of the first particle.
        m2 : float
            The mass of the second particle.
        m3 : float 
            The mass of the third particle.
        g : np.ndarray
            A 3x3 rotation matrix from the symmetry group.
        state : int
            An index slice for n_arr.
    
        Returns
        -------
        proj_matrix : np.ndarray
            A square matrix made up from the projectors for both the identical and non-identical case.
    
        '''
        m1 = self.masses[0]
        m2 = self.masses[1]
        m3 = self.masses[2]
        E_nSQ_arr, counts, n_arr = self.get_E_nSQ_vecs(Emax, nP, Lmax, nvec_cutoff , m1, m2, m3, False)
        E_nSQ_arr_identical, counts_identical, n_arr_identical = self.get_E_nSQ_vecs(Emax, nP, Lmax, nvec_cutoff , m1, m2, m3, True)
        n_arr_sorted = self.sort_by_rot(n_arr[state])
        n_arr_iden_sorted = self.sort_by_rot(n_arr_identical[state])
        ind_rep = self.get_induced_rep(n_arr_sorted, g, False)
        ind_rep_identical = self.get_induced_rep(n_arr_iden_sorted, g, True)
        proj = block_diag(ind_rep, ind_rep_identical)
        
        proj_matrix = np.zeros((len(proj), len(proj)))
    
        dim = 3
        for i in range(dim):
            start = i*(2*dim) 
            end = (i+1)*(2*dim)
            proj_matrix[start+i:end+i] = proj[start:end]
            proj_matrix[end+i] = proj[-dim+i]
        return proj_matrix
    


    def generic_zero_elimination(self, array, LIMIT=1.0e-13):
        '''
        
    
        Parameters
        ----------
        array : comple/np.ndarray
            A number or an array of elements.
            
        LIMIT : float
            The NVEC_CUTOFF limit for the zero elemination. The default is 1.0e-14.
    
        Returns
        -------
        array : complex/ np.ndarray
            The input parameter in returend with:
                Re(array)   if      All |Im(array)| < LIMIT
                Im(array)   if      All |Re(array| < LIMIT
                array[i] = 0.0  if     array[i] < LIMIT
                
    
        '''
        if isinstance(array, complex):
           if abs(array.imag) < LIMIT:
               array = array.real
           if abs(array.real) < LIMIT:
               array = array.imag*1j
           if (abs(array.real) < LIMIT) and (abs(array.imag) < LIMIT):
               array = 0.0         
        if isinstance(array, np.ndarray):
            array = np.where(abs(array) < LIMIT, 0.0, array)
            if np.where(abs(array.imag) < LIMIT) != 0.0:
                array = np.where(abs(array.imag) < LIMIT, array.real+0.0*1j, array)
            if np.where(abs(array.real) < LIMIT) != 0.0:
                array = np.where(abs(array.real) < LIMIT, array.imag*1j+0.0, array)
            if np.all(abs(array.imag) < LIMIT):
                array = array.real
            if np.all(abs(array.real) < LIMIT):
                array = array.imag
        return array


    def get_EigenValue_for_irrep(self, Emax, nP, Lmax, nvec_cutoff , masses, irrep, state):
        '''
        
        Parameters
        ----------
        n_arr : np.ndarray
            An array of size 3x3 representing the 3-vector [n1,n2,n3] 
            with a dimension of all possible combination for [n1SQ, n2SQ, n3SQ].
            
        irrep : str
            A string for an irrep of the octahedral group.
    
        Returns
        -------
        eig_values : np.darray
            An array carrying the eigen values of the projector of the induced irreps
            with dimension of 'irrep'.
    
        '''
        irrep_coeff = Group.bTdict['OhP'+'_'+irrep]
        dim = len(irrep_coeff)
        proj = dim*[[]]
        for n in range(dim):
            proj_tmp = 0.0
            for i in range(len(OhP)):
                irrep_tmp = self.get_total_proj(Emax, nP, Lmax, nvec_cutoff , masses, OhP[i], state)
                product  = irrep_coeff[n][i]*irrep_tmp
                proj_tmp = proj_tmp + product
            proj[n] = proj_tmp
        final_proj = np.array(proj)
        
        second_change_of_basis = block_diag(*(3*[C_MAT]))
        final_proj = self.generic_zero_elimination(second_change_of_basis@final_proj@(second_change_of_basis.T))
        
        
        eig_values, eig_vectors = np.linalg.eig(final_proj)
        eig_values = self.generic_zero_elimination(eig_values)
        eig_values_nonzero = eig_values[np.nonzero(eig_values)]
        return final_proj, eig_values_nonzero

    def get_EigenValue_for_irrep_single_isospin(self, Emax, nP, Lmax, nvec_cutoff , masses, irrep, state, total_isospin):
        
        irrep_coeff = Group.bTdict['OhP'+'_'+irrep]
        dim = len(irrep_coeff)
        proj = dim*[[]]
        for n in range(dim):
            proj_tmp = 0.0
            for i in range(len(OhP)):
                irrep_tmp = self.get_total_proj(Emax, nP, Lmax, nvec_cutoff ,masses, OhP[i], state)
                product  = irrep_coeff[n][i]*irrep_tmp
                proj_tmp = proj_tmp + product
            proj[n] = proj_tmp
        final_proj = np.array(proj)
        
        second_change_of_basis = block_diag(*(3*[C_MAT]))
        final_proj = second_change_of_basis@final_proj@(second_change_of_basis.T)
        #print(generic_zero_elimination(final_proj[0]))
        if total_isospin == 3:
            iso_list = [1,0,0,0,0,0,0]
        if total_isospin == 2:
            iso_list = [0,1,1,0,0,0,0]
        if total_isospin == 1:
            iso_list = [0,0,0,1,1,1,0]
        if total_isospin == 0:
            iso_list = [0,0,0,0,0,0,1]
    
        final_proj = block_diag(*(3*[block_diag(*(iso_list))]))@final_proj\
            @ block_diag(*(3*[block_diag(*(iso_list))]))
       
        eig_values, eig_vectors = np.linalg.eig(final_proj)
        eig_values = self.generic_zero_elimination(eig_values)
        eig_values_nonzero = eig_values[np.nonzero(eig_values)]
        return eig_values_nonzero
    
    def _get_summary(self, Emax, nP, Lmax, masses, nvec_cutoff, state):
        strtmp = '\n'
        irrep_set = ampyl.Irreps(nP).set
        for total_iso in [3, 2, 1, 0]:
            print('\n for isospin:', total_iso)
            for irrep in irrep_set:
                eig_values = self.get_EigenValue_for_irrep_single_isospin(Emax, nP, Lmax, nvec_cutoff , masses, irrep, state, total_iso)
                strtmp += f" eigenvalues of {eig_values} with length of {len(eig_values)} for Irrep {irrep} \n" 
        return strtmp
        
        
        
    
    def __str__(self):
        Emax = self.Emax
        nP = self.nP
        Lmax = self.Lmax
        masses = self.masses
        #nvec_cutoff = self.get_nvec_cutoff_n3zero(Emax, nP, Lmax, masses)
        nvec_cutoff = 1
        state = 1
        summary = self._get_summary(Emax, nP, Lmax, masses, nvec_cutoff, state)
        return summary

masses = 3*[1]
TWOPI = 2*np.pi
Lmax = 5.0
nP = np.array([0,0,0])
Emax = 7.0
m1 = m2 = m3 = 1.0
nvec_cutoff = 1
irrep_set = ampyl.Irreps(nP).set
state = 1 

nonint_proj = Nonint_Proj(Emax, nP, Lmax, masses)
cutoff = nonint_proj.get_nvec_cutoff_n3zero(Emax, nP, Lmax, masses)
#val_irrep = nonint_proj.get_EigenValue_for_irrep_single_isospin(Emax, nP, Lmax, nvec_cutoff , masses, 'A1PLUS', 1, 3)
print(nonint_proj)


# irrep_proj = []
# for irrep in irrep_set:
#     final_proj, eig_values = nonint_proj.get_EigenValue_for_irrep(Emax, nP, Lmax, nvec_cutoff , masses, irrep, state)
#     if len(eig_values) != 0:
#         irrep_proj = irrep_proj +[final_proj]
#         print(f"Irrep {irrep} has eigenvalues of {eig_values} with length of {len(eig_values)}")


class Test_group_theory(unittest.TestCase):
    """ """
    
    def result(self):
        pass

class Template(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Exectue set-up."""
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def __example(self, x):
        return x

    def test(self):
        """Example test."""
        self.assertEqual(10., self.__example(10.))