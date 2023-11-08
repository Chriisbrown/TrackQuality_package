
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
#import mplhep as hep
#plt.style.use(hep.style.CMS)

def cylinder(ax,r):
    u = np.linspace(0,2*np.pi,20)
    h = np.linspace(-115,115,2)
    x = np.outer(r*np.sin(u),np.ones(len(h)))
    y = np.outer(r*np.cos(u),np.ones(len(h)))
    z = np.outer(np.ones(len(u)),h)

    ax.plot_surface(x,y,z,alpha=0.05,color='k')

def disk(ax,Z):

    p = Circle((0, 0), 110,alpha=0.05,color='k')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=Z, zdir="z")

    #art3d.pathpatch_2d_to_3d(p, z=-Z, zdir="x")

def plot_helix(event):

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        figrphi,axrphi = plt.subplots(1,1,figsize=(8,8))
        figrz,axrz = plt.subplots(1,1,figsize=(30,8))

        for ievt in range(len(event["trk_pt"])):
            eta = event["trk_eta"][ievt]
            
            phi0 = event["trk_phi"][ievt]
            z0 = event["trk_z0"][ievt]
            pt = event["trk_pt"][ievt]

            rho = 100*(pt*1e9)/(3.8112*(3e8))
            theta = abs(np.sinh(eta))
            n = 10000

            Z = [z0]
            X = [0]
            Y = [0]
            R = [0]
            phis = [phi0]


            if event["trk_fake"][ievt] == 0:
                c = 'red'
                a = 0.1
            if event["trk_fake"][ievt] == 1:
                c = 'green'
                a = 1.0
            if event["trk_fake"][ievt] == 2:
                c = 'blue'
                a = 0.1

            for i in range(1,n):
                phi = phis[i-1] + np.pi/n
                x = X[i-1] + rho*np.sin(phi)*np.cos(phi)
                y = Y[i-1] + rho*np.sin(phi)*np.sin(phi)
                z = Z[i-1] + np.sign(eta)*np.sqrt(x**2+y**2)/np.tan(theta)
                
                X.append(x)
                Y.append(y)
                Z.append(z)
                R.append(np.sqrt(x**2+y**2))
                phis.append(phi)

                #if (event["trk_fake"][ievt] == 1):
                #print(rho,phi,eta,z0,X,Y,Z)

                if np.abs(z) > 280:
                    #Z.pop(i)
                    #X.pop(i)
                    #Y.pop(i)
                    #R.pop(i)
                    break
                if np.sqrt(x**2+y**2) > 120:
                    #Z.pop(i)
                    #X.pop(i)
                    #Y.pop(i)
                    #R.pop(i)
                    break

            ax.plot(X,Y,Z,color=c,alpha=a)

            axrphi.plot(Y,X,color=c,alpha=a)
            axrphi.set_xlim(-120,120)
            axrphi.set_ylim(-120,120)
            axrphi.set_xlabel("X")
            axrphi.set_ylabel("Y")

            barrel1 = plt.Circle((0, 0), 22, linewidth=1, facecolor = None, alpha=0.05,color='k')
            barrel2 = plt.Circle((0, 0), 38, linewidth=1, facecolor = None, alpha=0.05,color='k')
            barrel3 = plt.Circle((0, 0), 50, linewidth=1, facecolor = None, alpha=0.05,color='k')
            barrel4 = plt.Circle((0, 0), 68, linewidth=1, facecolor = None, alpha=0.05,color='k')
            barrel5 = plt.Circle((0, 0), 90, linewidth=1, facecolor = None, alpha=0.05,color='k')
            barrel6 = plt.Circle((0, 0), 108, linewidth=1, facecolor = None, alpha=0.05,color='k')

            axrphi.add_patch(barrel1)
            axrphi.add_patch(barrel2)
            axrphi.add_patch(barrel3)
            axrphi.add_patch(barrel4)
            axrphi.add_patch(barrel5)
            axrphi.add_patch(barrel6)


            axrz.plot(Z,R,color=c,alpha=a)
            axrz.set_xlim(-300,300)
            axrz.set_ylim(-120,120)
            axrz.set_xlabel("z")
            axrz.set_ylabel("r")

            axrz.plot([-120,120],[22,22],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-22,-22],linewidth=1, alpha=0.05,color='k')

            axrz.plot([-120,120],[38,38],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-38,-38],linewidth=1, alpha=0.05,color='k')

            axrz.plot([-120,120],[50,50],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-50,-50],linewidth=1, alpha=0.05,color='k')

            axrz.plot([-120,120],[68,68],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-68,-68],linewidth=1, alpha=0.05,color='k')

            axrz.plot([-120,120],[90,90],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-90,-90],linewidth=1, alpha=0.05,color='k')

            axrz.plot([-120,120],[108,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-120,120],[-108,-108],linewidth=1, alpha=0.05,color='k')

            axrz.plot([135,135],[22,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([135,135],[-22,-108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-135,-135],[22,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-135,-135],[-22,-108],linewidth=1, alpha=0.05,color='k')

            axrz.plot([160,160],[22,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([160,160],[-22,-108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-160,-160],[22,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-160,-160],[-22,-108],linewidth=1, alpha=0.05,color='k')

            axrz.plot([190,190],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([190,190],[-38,-108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-190,-190],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-190,-190],[-38,-108],linewidth=1, alpha=0.05,color='k')

            axrz.plot([220,220],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([220,220],[-38,-108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-220,-220],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-220,-220],[-38,-108],linewidth=1, alpha=0.05,color='k')

            axrz.plot([265,265],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([265,265],[-38,-108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-265,-265],[38,108],linewidth=1, alpha=0.05,color='k')
            axrz.plot([-265,-265],[-38,-108],linewidth=1, alpha=0.05,color='k')

            #if ievt > 0:
            #    break
        cylinder(ax,22)
        cylinder(ax,38)
        cylinder(ax,50)
        cylinder(ax,68)
        cylinder(ax,90)
        cylinder(ax,108)
        disk(ax,135)
        disk(ax,160)
        disk(ax,190)
        disk(ax,220)
        disk(ax,265)

        disk(ax,-135)
        disk(ax,-160)
        disk(ax,-190)
        disk(ax,-220)
        disk(ax,-265)

        ax.set_xlim(-120,120)
        ax.set_ylim(-120,120)
        ax.set_zlim(-300,300)


        '''

        ax.scatter(temp_df["trk_pt"],temp_df["trk_phi"],temp_df["trk_z0"],c=temp_df["trk_genuine"],cmap='bwr')
        ax.set_xlabel("trk_pt")
        ax.set_ylabel("trk_phi")
        ax.set_zlabel("trk_z0")
        plt.show()
        ax.scatter(temp_df["trk_pt"],temp_df["trk_chi2"],temp_df["trk_bendchi2"],c=temp_df["trk_genuine"],cmap='bwr')
        ax.set_xlabel("trk_pt")
        ax.set_ylabel("trk_chi")
        ax.set_zlabel("trk_bendchi")
        plt.show()
        '''
        #ax.scatter(temp_df["trk_nstub"],temp_df["trk_lhits"],temp_df["trk_dhits"],c=temp_df["trk_genuine"],cmap='bwr')
        #ax.set_xlabel("trk_nstub")
        #ax.set_ylabel("trk_lhits")
        #ax.set_zlabel("trk_dhits")
        #plt.axis('off')
        plt.show()



