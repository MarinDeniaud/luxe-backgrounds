import pymad8
import pymadx
import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd

class Plot :
	'''Class to load pymad8 and pymadx DataFrames and plot the comparison 
	of the two model with respect to different parameters :
	plot_data = compareMad8MadX.Plot('/mad8_twiss.tape','/madx_twiss.tape')
	plot_data.plotBetas()

	plot avaliable are plotBetas, plotAlphas, plotMus, plotDisp and plotSigmas '''
	#'../01_mad8/TWISS_CL_T20'
	#'../02_madx/twiss-t20-with-ff-ff-matched.tfs'
	def __init__(self,mad8file,madxfile):
		'''Load from form the two filenames both the twiss files and 
		the twiss DataFrame and save them as internal variables.
		Also match the positions from the tow lines with respect to the LUXE IP'''
		r = pymad8.Output.OutputReader()
		[c,self.twiss_mad8] = r.readFile(mad8file)
		self.df_twiss_mad8 = pymad8.OutputPandas(mad8file)
		
		self.twiss_madx = pymadx.Data.Tfs(madxfile)
		self.df_twiss_madx = _pd.DataFrame(self.twiss_madx.data,index=self.twiss_madx.columns).transpose()

		S_IP_mad8 = self.df_twiss_mad8.getElementByNames('IP.LUXE.T20','S')
		S_IP_madx = self.df_twiss_madx.loc[self.df_twiss_madx['NAME']=='LUXE.IP','S']
		Delta_S = S_IP_mad8 - S_IP_madx
		self.df_twiss_madx['S'] = self.df_twiss_madx['S'].apply(lambda x:x+Delta_S)

	def plotBetas(self):
		'''Plot the Beta functions for both planes and both Mad'''
		#self.df_twiss_mad8.data.plot('S',['BETX','BETY'],subplots=True)
		#self.df_twiss_madx.plot('S',['BETX','BETY'],subplots=True)

		_plt.figure(1,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['BETX'], label='Beta_x Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['BETX'], label='Beta_x MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['BETY'], label='Beta_y Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['BETY'], label='Beta_y MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Beta [m]')
		_plt.legend()

		_plt.show()

	def plotAlphas(self):
		'''Plot the Alpha functions for both planes and both Mad'''
		_plt.figure(1,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['ALPHX'], label='Alpha_x Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['ALFX'], label='Alpha_x MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['ALPHY'], label='Alpha_y Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['ALFY'], label='Alpha_y MadX')
                
		_plt.xlabel('S [m]')
		_plt.ylabel('Alpha [rad]')
		_plt.legend()

		_plt.show()

	def plotMus(self):
		'''Plot the Mu functions for both planes and both Mad'''
		_plt.figure(1,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['MUX'], label='Mu_x Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['MUX'], label='Mu_x MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['MUY'], label='Mu_y Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['MUY'], label='Mu_y MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Mu [?]')
		_plt.legend()

		_plt.show()

	def plotDisp(self):
		'''Plot the Dispertion functions for both planes and both Mad'''
		_plt.figure(1,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['DX'], label='Disp_x Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['DX'], label='Disp_x MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['DY'], label='Disp_y Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['DY'], label='Disp_y MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Disp [m]')
		_plt.legend()

		_plt.figure(2,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['DPX'], label='Disp_xp Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['DPX'], label='Disp_xp MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['DPY'], label='Disp_yp Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['DPY'], label='Disp_yp MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Disp_p [rad]')
		_plt.legend()

		_plt.show()

	def plotSigmas(self):
		'''Plot the beam size and beam divergence functions for both planes and both Mad'''
		self.df_twiss_mad8.calcBeamSize(3.58*10**-11,3.58*10**-11,1*10**-6)
		
		_plt.figure(1,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['SIGX'], label='Sigma_x Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['SIGMAX'], label='Sigma_x MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['SIGY'], label='Sigma_y Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['SIGMAY'], label='Sigma_y MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Sigma [m]')
		_plt.legend()

		_plt.figure(2,figsize=(10,6))
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['SIGXP'], label='Sigma_xp Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['SIGMAXP'], label='Sigma_xp MadX')
		_plt.plot(self.df_twiss_mad8.data['S'],self.df_twiss_mad8.data['SIGYP'], label='Sigma_yp Mad8')
		_plt.plot(self.df_twiss_madx['S'],self.df_twiss_madx['SIGMAYP'], label='Sigma_yp MadX')

		_plt.xlabel('S [m]')
		_plt.ylabel('Sigma_p [rad]')
		_plt.legend()

		_plt.show()

	def _plotForEachMad(self):
		pass
