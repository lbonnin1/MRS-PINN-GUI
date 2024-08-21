import re
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
import mplcursors
from toolbar import ToolBar
from control import ControlBar
import numpy as np
from PyQt5.QtGui import QKeySequence


class MainWindow(QMainWindow):
    """Main window for the spectroscopy data analysis application.

    This window provides a user interface for loading, visualizing, and analyzing spectroscopy data.
    It includes controls for adjusting parameters such as width and amplitude of spectral peaks, selecting different types of peaks, and exporting data.

    Attributes:
        ppm_data (numpy.ndarray):Array of PPM data loaded from Basisset CSV file.
        basisset_data (numpy.ndarray):Array of basisset data loaded from Basisset CSV file.
        loaded_data: Placeholder for loaded spectroscopy data.
        file_path (str): Path to the loaded data file.
        metabolites (list): List of metabolites to be displayed and analyzed.
        largeur_mh (dict): Dictionary to store metabolite widths at half maximum amplitude.
        amplitude_values (dict): Dictionary to store metabolite amplitude values.
        center_values (list): List of default center values in ppm for each metabolite.
        width_values (dict): Dictionary to store metabolite width values.
        metab_visibility_states (dict): Dictionary to store visibility states of metabolites.
        type_values (dict): Dictionary to store the curve type values of metabolites.
        canvas (FigureCanvas): Canvas for embedding Matplotlib figure.
        fig (matplotlib.figure.Figure): Matplotlib figure for plotting.
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting spectroscopy data.
        tool_bar (ToolBar): Toolbar for importing and exporting data.
        control_bar (ControlBar): Control bar for adjusting parameters and selecting metabolites.
    """

    def __init__(self):
        """Initialize the MainWindow.

        Sets up the UI elements, including the central canvas for plotting,
        the control bar for adjusting parameters, and the toolbar for importing
        and exporting data.
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """Set up the user interface.
        This method initializes the main window layout.
        It includes the central canvas for plotting spectroscopy data, the control bar for adjusting parameters and the toolbar for importingand exporting data.
        """
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # definition of consts
        self.ppm_data = [] #values of ppm data to be extracted from the basis csv file and stored here
        self.loaded_data = None #spectra data in nupmy format or dicom preprocesses to be stored here
        self.file_path = None #file path to the spectra data file to be stored here
        self.colors = {
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
            "Orange": (255, 165, 0),
            "Purple": (128, 0, 128),
            "Pink": (255, 192, 203),
            "Brown": (165, 42, 42),
            "Lime": (10, 135, 0),
            "Indigo": (75, 0, 130),
            "Violet": (238, 130, 238),
            "Gold": (255, 215, 0),
            "Silver": (192, 192, 192),
            "Maroon": (128, 0, 0),
            "Olive": (128, 128, 0),
            "Teal": (0, 128, 128),  
            "Sky Blue": (135, 206, 235),  
            "Turquoise": (64, 224, 208), 
            "Slate Gray": (112, 128, 144),  
            "Salmon": (250, 128, 114),  
            "Chocolate": (210, 105, 30), 
            "Crimson": (220, 20, 60),  
            "Plum": (221, 160, 221),  
            "Sea Green": (46, 139, 87),  
            "Dark Orchid": (153, 50, 204)  
        }


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #lists and dicts of metabolites and parameters initialization
        #self.metabolites = ['Pcr + Cr','Glu+ Gln','Pcho+GPC','Ins+tau','Pcr+Cr 2','Glu+Gln 2','Ala 1', 'Ala 2', 'Asp 1', 'Asp 2', 'Asp 3', 'Cr 1', 'Cr 2', 'PCr 1', 'PCr 2', 'Glc 1', 'Glc 2','Glc 3','Glu 1','Glu 2','Glu 3','Gln 1','Gln 2','Gln 3','GPC 1','GPC 2', 'GPC 3', 'Ins 1', 'Ins 2', 'Lac 1', 'NAA 1', 'NAA 2', 'NAA 3','NAAG 1', 'NAAG 2', 'NAAG 3','NAAG 4','NAAG 5','PCh 1','PCh 2','Tau 1','Tau 2', 'Scyllo 1', 'GABA 1','GABA 2', 'GABA 3','GSH 1', 'GSH 2','GSH 3', 'GSH 4']
        #self.center_values =[3.92,3.745, 3.2,3.25, 3.02, 2.085, 3.77, 1.457, 3.887, 2.8, 2.641, 3.907, 3.022, 3.905, 3.02, 3.78, 3.438, 3.22, 3.738, 2.34, 2.077, 3.748, 2.432, 2.10, 3.90, 3.636, 3.20, 3.57, 3.26, 1.305, 2.67, 2.475, 2.00, 2.71, 2.506, 2.18, 2.034, 1.869, 3.637, 3.203, 3.417, 3.238, 3.334, 3.006, 2.276, 1.88, 3.764, 2.942, 2.515, 2.142]
        self.metabolites = ["PCr + Cr 2","GSH + Glu + Gln","MIns 2","MIns 1","PCh + GPC","PCr + Cr 1","Asp","NAA 2","Gln","Glu","NAA 1","Lac","Lip"]
        self.center_values= [ 3.92, 3.768 ,3.615,3.52 ,3.21 ,3.02,2.8,2.57,2.43,2.34 ,2,1.3,0.9]
        #self.width_values = {metabolite: 1 for metabolite in self.metabolites}
        self.metab_visibility_states = {metabolite: False for metabolite in self.metabolites}
        self.type_values = {metabolite: "Lorentzienne" for metabolite in self.metabolites}
        self.largeur_mh={metabolite:0.15 for metabolite in self.metabolites}
        self.amplitude_values = {metabolite: 75000 for metabolite in self.metabolites}
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def process_metabolites(metabolites):
            processed_metabolites = [re.sub(r'\d+$', '', metabolite).strip() for metabolite in metabolites]
        
            for i, metabolite in enumerate(processed_metabolites):
                if metabolite == 'MIns':
                    processed_metabolites[i] = 'Ins'
            return processed_metabolites
        self.metab_pro = process_metabolites(self.metabolites)
        #print(self.metab_pro)
        #setting the window parametrs
        self.setWindowTitle("spectro")
        self.setGeometry(400, 400, 800, 600)

        # Styling with QSS
        with open("style.qss", "r") as f:
            self.setStyleSheet(f.read())

        # creating the main layout 
        central_widget = QWidget()  
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget) 

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #creating a plotting canvas 
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_title('Ajustement spectral 1H-SRM 7T')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('PPM')
        #setting the axes limits
        self.ax.set_xlim(0,4)
        self.ax.set_ylim(0,150000)
        #inverting the ax to go from right to left
        self.ax.invert_xaxis()
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #creating a toolbar from the toolbar class where the menu component are defined
        self.tool_bar = ToolBar(self)
        self.addToolBar(self.tool_bar)
        self.save_shortcut = QShortcut(QKeySequence.Save, self)
        self.save_shortcut.activated.connect(lambda: self.tool_bar.export("json"))
        self.open_shortcut = QShortcut(QKeySequence.Open, self)
        self.open_shortcut.activated.connect(lambda: self.tool_bar.load())

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # creating a control section from the control class which groups all the metabolite control component and the basis set reference control components
        self.control_bar = ControlBar(self)

        #adding the component to the main layout
        layout.addWidget(self.canvas, stretch=6)
        layout.addWidget(self.control_bar, stretch=4) 
      
        #Full screen window
        self.showMaximized() 
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    def update_plot(self):
        """
        Update the spectroscopy plot based on current parameter values.

        This method updates the spectroscopy plot with the latest parameter values (amplitude, width, type, and visibility).
        It iterates over the selected metabolites, calculates the corresponding spectral data based on the selected type of peak (Lorentzian or Gaussian), and plots the data.
        Additionally, it computes the total spectral data by summing the contributions from all selected metabolites and plots the sum as well.

        This method is called whenever there is a change in parameter values or metabolite selections that affect the appearance of the spectroscopy plot.
        """
        # Store current xlim and ylim for interactive zoom
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Clear the axis
        self.ax.clear()

        # Plot loaded spectra data and basisset reference spectra data
        self.tool_bar.plot_loaded_data()
        self.control_bar.plot_basisset()

        # Initialize teh global fit to zero
        total_y = np.zeros_like(self.ppm_data)
        diff= np.zeros_like(self.ppm_data)
        
        # Iterate over metabolites
        for metabolite in self.metabolites:
            # Check if the metabolite is visible
            if self.metab_visibility_states[metabolite]:

                #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # Get parameter values for the metabolite
                amplitude = self.amplitude_values[metabolite]
                width = self.largeur_mh[metabolite]
                center = self.center_values[self.metabolites.index(metabolite)]
                type = self.type_values[metabolite]
                #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                # Calculate y values based on the peak type using the parameter values
                if type == "Lorentzienne":
                    y = amplitude / (1 + ((self.ppm_data - center) / (width/2) ) ** 2)
                else:
                    y = amplitude * np.exp(-4*np.log2(2) * ((self.ppm_data - center) / (width))**2)
            

                # Calculate the width at half maximum amplitude
                #amplitude_a_mi_hauteur = amplitude / 2.0
                #indices_a_mi_hauteur = np.where(y >= amplitude_a_mi_hauteur)[0]
                #if len(indices_a_mi_hauteur) > 0:  
                #   self.largeur_mh[metabolite] = self.ppm_data[indices_a_mi_hauteur[-1]] - self.ppm_data[indices_a_mi_hauteur[0]] 
                    

                # Update global fit by taking element-wise maximum
                total_y += y
                 # error estimation metrics : scr, mse, mape, r^2

                """for i in range(8192):    
                    diff[i]=(self.loaded_data[i]-total_y[i])
                    scr+=diff[i]**2
                    value_mape+=np.abs(diff[i]/self.loaded_data[i])
                mape=value_mape/8192
                mse=scr/8192
                print(mse)
                print(mape)
                """
                diff = self.loaded_data - total_y
                mse = np.mean(diff**2)
                non_zero_mask = self.loaded_data != 0
                mape = np.mean(np.abs(diff[non_zero_mask] / self.loaded_data[non_zero_mask]))
                rss =np.sum(diff**2)
                tss = np.sum((self.loaded_data - np.mean(self.loaded_data))**2)
                r_squared = 1 - (rss / tss)

                #print(f"RSS: {rss}")
                #print(f"MSE: {mse}")
                #print(f"MAPE: {mape}")
                #print(f"R-squared: {r_squared}")
                #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # Plot the fit curves
                color = self.control_bar.metab_checkboxes[metabolite].color
                normalized_color = tuple(c / 255 for c in color)
                self.ax.plot(self.ppm_data, y,color=normalized_color)
            
        # Plot the global spectra fit
        self.ax.plot(self.ppm_data, total_y, label="global fit", linestyle="--", color="red")
        #self.ax.plot(self.ppm_data, diff,color="#888")
 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       
        # Set axis labels, title, legend, and limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_title('Ajustement spectral 1H-SRM 7T')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('PPM')
        self.ax.legend()   
        # Redraw canvas
        self.canvas.draw()

        # Push current state to toolbar for zoom
        self.tool_bar.toolbar.push_current()
        
     

 #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       

if __name__ == '__main__':
    """The program's main function
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

