"""
mbpytexlib.py contains auxiliary functions for easier work with pythontex documents.
"""

#############################################################################
# global functions
#############################################################################

# how to keep fraction in sympy output (not implemented yet):
# https://stackoverflow.com/questions/52402817/how-to-keep-the-fraction-in-the-output-without-evaluating

def round_expr(expr, num_digits):
    # how to print sympy numbers rounded to n decimals:
    # https://stackoverflow.com/questions/48491577/printing-the-output-rounded-to-3-decimals-in-sympy
    
    import sympy as sp
    return expr.xreplace(
            {n : round(n, num_digits) for n in expr.atoms(sp.Number)})

def matrix(M, mtype='matrix', dec=5):
    """Return the latex representation of sympy or numpy matrix"""
    # types of matrices can be found here:
    # https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix
    
    import sympy as sp
    
    S = sp.Matrix(M) # converting to sympy Matrix in case M is a numpy matrix
    
    S = round_expr(S, dec) # rounding to specific number of decimals
        
    latexStr =  sp.latex(S)

    if mtype!='matrix':
        latexStr = latexStr.replace('matrix',mtype)

    latexStr = latexStr.replace('\\left[','')
    latexStr = latexStr.replace('\\right]','')
    
    return latexStr


def pmatrix(M, dec=5):
    """Return the latex representation of sympy or numpy matrix"""
    # types of matrices can be found here:
    # https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix
    
    return matrix(M,mtype='pmatrix',dec=dec)
    
def bmatrix(M, dec=5):
    """Return the latex representation of sympy or numpy matrix"""
    # types of matrices can be found here:
    # https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix
    
    return matrix(M,mtype='bmatrix',dec=dec)

def m2l(M):
    """Wrapper function with short name"""
    import numpy as np
    
    if hasattr(M, "__len__"):
        # numpy or sympy matrix
        ret = bmatrix(M)
    else:    
        # if scalar or sympy symbol
        ret = str(M)        
        
    return ret


def setTeXLikeFonts(plt):
    """ LaTeX-style fonts in pyplots"""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=10.0)
    plt.rc('legend', fontsize=10.0)
    plt.rc('font', weight='normal')


if __name__ == "__main__":
    
    # some tests
    import numpy as np
    import sympy as sp
    
    N = np.array([[1/3,2,1],
                   [0,3,1],
                   [2,1,1]])
    
    k = sp.symbols("k")
    
    S = sp.Matrix([[k,k,1/3],
                   [0,k,1],
                   [2,1,k]])
    
    
    print(matrix(N))
    print(pmatrix(S, 3))
    print(m2l(N))

# version 2

#def npmatrix(M, type='matrix'):
#    """Return the latex representation of the numpy matrix"""
#    # types of matrices can be found here:
#    # https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix
#    lines = str(M).replace('[', '').replace(']', '').splitlines()
#    rv = [r'\begin{' + type + r'}']
#    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#    rv += [r'\end{' + type + r'}']
#    return '\n'.join(rv)
#
#
#def npbmatrix(M):
#    """Return the latex representation of bmatrix"""
#    return npmatrix(M, type='bmatrix')    
#
#
#def nppmatrix(M):
#    """Return the latex representation of bmatrix"""
#    return npmatrix(M, type='pmatrix')    
#
#
#
#def spmatrix(M, type='matrix'):
#    """Return the latex representation of sympy matrix"""
#    # types of matrices can be found here:
#    # https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix
#    
#    import sympy as sp
#    latexStr =  sp.latex(M)
#    latexStr = latexStr.replace('matrix',type)
#    
#    return latexStr
#
#def spbmatrix(M):
#    """Return the latex representation of bmatrix"""
#    return spmatrix(M, type='bmatrix')    
#
#
#def sppmatrix(M):
#    """Return the latex representation of bmatrix"""
#    return spmatrix(M, type='pmatrix')    








# version 1


#def vector(v, separator, type='matrix'):
    #elements = str(v).replace('[', '').replace(']', '').split()
    #rv = [r'\begin{' + type + r'}']
    #rv += [ (' '+separator+' ').join(elements) ]
    #rv += [r'\end{' + type + r'}']
    #return ' '.join(rv)
    

#def colvector(v, type='bmatrix'):
    #"""Return the latex representation of column vector"""
    #return vector(v, separator=r'\\', type=type)
    

#def rowvector(v, type='bmatrix'):
    #"""Return the latex representation of row vector"""
    #return vector(v, separator=r'&', type=type)



#def Lprint(latexinput, block='$$'):
    #"""Return a string with full latex surrounded by block"""

    #if isinstance(latexinput, Sequence):
        #latexstr = ''.join(map(str,latexinput))
    #elif isinstance(latexinput, six.string_types):
        #latexstr = latexinput
    #else:
        #latexstr = str(latexinput)

    #Surround with correct latex scope
    #if block == '$$' or block == '$':
        #begin = block
        #end = block
    #elif block == r'\[':
        #begin = block
        #end = r'\]'
    #else:
        #begin = r'\begin{' + block + r'}'
        #end = r'\end{' + block + r'}'

    #return Latex(begin + latexstr + end)
