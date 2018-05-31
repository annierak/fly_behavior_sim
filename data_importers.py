import h5py
import json
import scipy
import itertools
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import sys
from pompy.models import PlumeStorer
import dill as pickle
import pompy.processors as processors

class ImportedPlumes(object):
    def __init__(self,pkl_file,array_z,array_dim_x,array_dim_y,puff_mol_amount):
        with open(pkl_file,'r') as f:
            self.plumeStorer = pickle.load(f)
        self.array_gen = processors.ConcentrationValueCalculator(puff_mol_amount)

    def puff_array_at_time(self,t):
        ind = int(scipy.floor(t/self.plumeStorer.dt_store))
        array_end = int(self.plumeStorer.puff_array_ends[ind])
        return self.plumeStorer.big_puff_array[ind,0:array_end,:]

    def value(self,t,xs,ys):
        puff_array = self.puff_array_at_time(t)
        self.array_gen.calc_conc_list(puff_array, xs, ys, z=0)

class ImportedConc(object):
#Note that each array here has been stored in the order that produces the right
#image (x goes top to bottom, y left to right)
    def __init__(self,hdf5_file,cmap='Blues'):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        self.simulation_region = run_param['simulation_region']
        self.xmin,self.xmax,self.ymin,self.ymax = run_param['simulation_region']
        self.dt_store = run_param['dt_store']
        #concentration data stored here, x by y by t
        self.t_stop = run_param['simulation_time']
        self.cmap = cmap
        self.vmin,self.vmax = run_param['imshow_bounds']
        self.conc_array = self.data['conc_array']
        self.grid_size = scipy.shape(self.conc_array[0,:,:])

    def array_at_time(self,t):
        ind = scipy.floor(t/self.dt_store)
        return self.conc_array[ind,:,:]

    def plot(self,t):
        array_at_time = self.array_at_time(t)
        image=plt.imshow(array_at_time, extent=self.simulation_region,
        cmap=self.cmap,vmin=self.vmin,vmax=self.vmax)
        return image

    def value(self,t,xs,ys):
        array_at_time = self.array_at_time(t)
        # plt.imshow(array_at_time, extent=self.simulation_region,
        # cmap=self.cmap,vmin=self.vmin,vmax=self.vmax)
        # print(scipy.shape(scipy.sum(array_at_time,0)))
        # print(scipy.where(scipy.sum(array_at_time,0)>0.01))
        # print(scipy.where(scipy.sum(array_at_time,1)>0.01))
        conc_list = scipy.zeros(scipy.size(xs))
        x_inds = scipy.floor(
        (xs-self.xmin)/(self.xmax-self.xmin)*self.grid_size[0])
        y_inds = scipy.floor(
        abs(ys-self.ymax)/(self.ymax-self.ymin)*self.grid_size[1])
        # print(x_inds,y_inds)
        # plt.show()
        for i,(x_ind,y_ind) in zip(range(len(xs)),zip(x_inds,y_inds)):
            try:
                conc_list[i] = array_at_time[y_ind,x_ind]
            except(IndexError):
                # print(x_ind,y_ind)
                conc_list[i]=0.
        return conc_list

    def get_image_params(self):
        return (self.vmin,self.vmax,self.cmap)

class ImportedWind(object):
    def __init__(self,hdf5_file):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        self.dt_store = run_param['dt_store']
        self.t_stop = run_param['simulation_time']
        self.x_points = scipy.array(run_param['x_points'])
        self.y_points = scipy.array(run_param['y_points'])
        self.velocity_field = self.data['velocity_field']
        self.evolving = True

    def quiver_at_time(self,t):
        ind = int(t/self.dt_store)
        velocity_field = self.velocity_field[ind,:,:,:]
        return scipy.array(velocity_field[:,:,0]),\
            scipy.array(velocity_field[:,:,1])

    def get_plotting_points(self):
        x_origins,y_origins = self.x_points,self.y_points
        coords = scipy.array(list(itertools.product(x_origins, y_origins)))
        x_coords,y_coords = coords[:,0],coords[:,1]
        return x_coords,y_coords

    def velocity_at_pos(self,t,x,y):
    #This will use the same interpolating method that pompy wind field uses
        us,vs = self.quiver_at_time(t)
        interp_u = interp.RectBivariateSpline(self.x_points,self.y_points,us)
        interp_v = interp.RectBivariateSpline(self.x_points,self.y_points,vs)
        # if interp_u(x,y)<0.001:
        #     print('messed up u')
        #     sys.exit()
        # if interp_v(x,y)<0.001:
        #     print('messed up v')
        #     sys.exit()
        return scipy.array([float(interp_u(x, y)),
                                 float(interp_v(x, y))])
    def value(self,t,x,y):
    #performs velocity_at_pos on an array of x,y coordinates
        if type(x)==scipy.ndarray:
            wind = scipy.array([
                self.velocity_at_pos(t,x[i],y[i]) for i in range(len(x))
                ])
            # z_index = scipy.where((wind[:,0]==0. | wind[:,1]==0.))
            # if scipy.size(z_index)>0:
            #     print(x[z_index],y[z_index])
            #     print('wind value problem')
            #     sys.exit()
            return wind[:,0],wind[:,1]
        else:
            return self.velocity_at_pos(t,x,y)
