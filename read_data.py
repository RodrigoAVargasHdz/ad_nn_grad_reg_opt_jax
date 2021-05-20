import os
import numpy as np

def read_data():
# 	READ CARTESIAN COORDINATES
	f = open('Yafu/nh3_disps/geom.all', 'r')
	X = np.zeros((1,12)) # geometris in the cartesian coordinates
	Lines = f.readlines()
	j = 0
	for i,l in enumerate(Lines):
		d = l.split()
		atom = d[0]
		mass = d[-1]
		if atom == 'N':
			r = np.array(d[2:-1],dtype=float)
		else:
			j += 1
			r = np.append(r,np.array(d[2:-1],dtype=float))
			if j == 3:
				X = np.vstack((X,r))
				j = 0
	f.close()
	X = X[1:,:]
	print(X.shape)		

# 	READ ENERGY
	y = np.loadtxt('Yafu/nh3_disps/energy.all')	
	print(y.shape)
# 	shift
	y = 56.475986184916 + y
	return X,y


def read_gradient_energy(i=1):
    f = open('Yafu/nh3_disps/cartgrd.drt1.state{}.all'.format(i),'r')
    Lines = f.readlines()

    j = 0
    d0 = []
    GX = np.zeros((1,12))
    for i,l in enumerate(Lines[:]):
        d = l.split()
        if len(d) > 0: # not an empty list
            j = j + 1
            for x in d:
                x0 = x.replace('D','E')
                d0.append(x0)
                 
        if j % 4 == 0 and j > 0:
            d0 = np.array(d0,dtype=float)
            GX = np.vstack((GX,d0))
            j = 0
            d0 = []
                    
    GX = GX[1:,:]
    return GX

def training_data(N):
    X = jnp.load('nh3_cgeom.npy')
    y = jnp.load('nh3_energy.npy')
    

def main():


    X,y = read_data()
    GX0 = read_gradient_energy(1)
    GX1 = read_gradient_energy(2)
    print(X.shape,y.shape,GX0.shape,GX1.shape)
    print(X[1,:])
    print(GX0[1,:])
    print(GX1[1,:])
    
#     print(GX0[-2:,:])
#     np.save('nh3_cgeom',X)
#     np.save('nh3_energy',y)
#     np.save('nh3_grad_energy_state_0',GX0)
#     np.save('nh3_grad_energy_state_1',GX1)
# 	np.sabe('nh3_grad_energy',GX)
	
	
if __name__ == "__main__":

    main()