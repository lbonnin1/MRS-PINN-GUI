import json
import os
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import mplcursors
import numpy as np
import xml.etree.ElementTree as ET
from pretraitement import *


class ToolBar(QToolBar):
    """ Generates a toolbar menu for the interface.

        Components:
        - Import button: connected to the load method to import npy files.
        - Export button: connected to the export method to export the data in XML format.
        - Navigation toolbar: matplotlib navigation toolbar.

        Methods:
        - export: Exports each metabolite's parameters from the plot in XML format iteratively.
        - load: Loads data of a spectra from a numpy extension file.
        - plot_loaded_data: Plots data of the spectra on the canvas.
        """
    def __init__(self, parent=None):
        """Initializes the toolbar with import and export buttons.

        Parameters:
            parent : QWidget, optional
                The parent widget. Defaults to None.

        Attributes:
            parent : QWidget
                The parent widget.
            button_import : QPushButton
                Button for importing data.
            button_export : QPushButton
                Button for exporting data.
            toolbar : NavigationToolbar2QT
                Matplotlib toolbar for the parent canvas.
        """
        super(ToolBar, self).__init__(parent)
        self.parent = parent  

        # Button for importing data
        button_import = QPushButton("Import spectra")
        button_import.clicked.connect(self.load) 
        self.addWidget(button_import)

        # Button for importing data
        button_import = QPushButton("Import fits")
        button_import.clicked.connect(self.load_fits) 
        self.addWidget(button_import)


        # Button for exporting data with dropdown menu
        dropdown_button = QPushButton()
        dropdown_button.setText("Export")
        dropdown_menu = QMenu(self)
        dropdown_button.setMenu(dropdown_menu)
        self.addWidget(dropdown_button)

        # Actions for exporting data in JSON and XML formats
        action_json = QAction("JSON", self)
        action_xml = QAction("XML", self)
        action_json.triggered.connect(lambda: self.export("json"))
        action_xml.triggered.connect(lambda: self.export("xml"))
        dropdown_menu.addAction(action_json)
        dropdown_menu.addAction(action_xml)

        #Matplotlib navigationToolbar for manipulating the plot
        self.toolbar = NavigationToolbar(self.parent.canvas, self)
        for widget in self.toolbar.findChildren(QWidget):
            widget.setStyleSheet(f'color:white;')
        self.addWidget(self.toolbar)

        

        


    def export(self,format):
        """Exports each metabolite's parameters from the plot in XML or JSON format iteratively.

        The exported file takes the name of the original annotated spectra.
        The generated file architecture
            - metabolites: groups all the metabolites. each contains sub-elements, its parameters.
                - metabolite: metabolite name
                    - width: Full width at half maximum amplitude.
                    - amplitude: concentration of the metabolite.
                    - center: frequency of resonance of the metabolite, measured in PPM.
                    - type: the curve type: Gaussian or Lorentzian.

        Parameters:
            self : Toolbar
                The Toolbar object.

        Returns:
            None
        """
       

        file_name = os.path.join(os.path.dirname(self.parent.file_path), os.path.splitext(os.path.basename(self.parent.file_path))[0])
        
        print(file_name)
        root = ET.Element("Metabolites")
        metabolites_data=[]
        try:
            if format == "xml":
                for metabolite in self.parent.metabolites:
                    #get parameters values for each metabolite
                    width = self.parent.largeur_mh[metabolite]
                    amplitude = self.parent.amplitude_values[metabolite]
                    center =  self.parent.center_values[self.parent.metabolites.index(metabolite)]
                    type=self.parent.type_values[metabolite]
                    #create XML structure
                    metabolite_element = ET.SubElement(root, "Metabolite", name=metabolite)
                
                    width_element = ET.SubElement(metabolite_element, "Width")
                    width_element.text = str(width)
                    
                    amplitude_element = ET.SubElement(metabolite_element, "Amplitude")
                    amplitude_element.text = str(amplitude)
                    
                    center_element = ET.SubElement(metabolite_element, "Center")
                    center_element.text = str(center)
                    
                    type_element = ET.SubElement(metabolite_element, "Type")
                    type_element.text = str(type)
                tree = ET.ElementTree(root)
                #write on xml file
                tree.write(file_name+'.xml')
            elif format== "json":
                #get parameters values for each metabolite
                for metabolite in self.parent.metabolites:
                    width = self.parent.largeur_mh[metabolite]
                    amplitude = self.parent.amplitude_values[metabolite]
                    center = self.parent.center_values[self.parent.metabolites.index(metabolite)]
                    type = self.parent.type_values[metabolite]
                    
                    # Create a dictionary for each metabolite
                    metabolite_data = {
                        "name": metabolite,
                        "Width": width,
                        "Amplitude": amplitude,
                        "Center": center,
                        "Type": type
                    }
                    
                    # Append the metabolite data to the list
                    metabolites_data.append(metabolite_data)
                #write on the json file
                json_file_path = file_name+'.json'
                with open(json_file_path, "w") as json_file:
                    json.dump(metabolites_data, json_file, indent=4)
            #display success message
            QMessageBox.information(self, "information", "Data exported successfully.")
        except Exception as e:
                QMessageBox.information(self, "information", "No data exported.")
                
 
    def load(self):
        """Opens a file dialog to select a numpy or dicom spectra data file and process it.

        This method prompts the user to select a file using a file dialog window. The supported file types are NumPy files (.npy) and DICOM files (.dcm).
        
        If a file is selected, the method attempts to load the data from the file. If the file is a DICOM file, it is preprocessed then loaded. NumPy files are directly loaded using the `np.load` function.

        After loading the data, the method resets default parameter values for each metabolite, including width, visibility state, type, largeur_mh, and amplitude. These default values are then used to update the plot with the newly loaded data.

        If an error occurs during the loading process, such as a corrupted file or unsupported file type, an error message is displayed to the user.

        Returns:
            None
        """
        # file filters for NumPy files and DICOM files
        file_filters = "spectra files (*.npy *.dcm)"

        # Prompt the user to select a file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filters)
                
        if file_path:
            try:
                if file_path.endswith('.dcm'):
                    # Process DICOM file
                    table_data, points, rows, columns = load_dcm_data(file_path)
                    tabledatatfreshape = tf_data(table_data, rows, columns, points)

                    self.parent.loaded_data = traitement(tabledatatfreshape, self.parent.ppm_data)
                else:
                    # Load data from the selected NumPy file
                    self.parent.loaded_data = np.load(file_path)
                
                # Initialize default parameter values for each metabolite
                for metab in self.parent.metabolites:
                    self.parent.metab_visibility_states[metab] = False
                    self.parent.type_values[metab] = "Lorentzienne"
                    self.parent.largeur_mh[metab] = 0.15
                    self.parent.amplitude_values[metab] = 75000.00
                    self.parent.control_bar.metab_checkboxes[metab].setChecked(False)
                    self.parent.control_bar.lorentzien_radiobuttons[metab].setChecked(True)
                    
                    self.parent.control_bar.width_sliders[metab].setValue(150)
                    self.parent.control_bar.amplitude_sliders[metab].setValue(750)

                # Update the plot with the newly loaded data and pparametrs curves
                self.parent.update_plot()
                
                # Store the file path to use for saving the results witha matching file name
                self.parent.file_path = file_path 

                # Display a success message to the user if the load was successfully done
                QMessageBox.information(self, "Information", "Spectra uploaded successfully.")
            except Exception as e:
                # Handle errors during the loading process
                print("Error loading file:", e)
                QMessageBox.information(self, "Information", "Spectra file corrupted.")


    
    def plot_loaded_data(self):
        """
        Plots data of the spectra from the loaded_data variable in the parent class.

        Parameters:
            
            self : Toolbar
                The Toolbar object.

        Returns:
            None
        """
        #the loaded data is the spectra data files
        if self.parent.loaded_data is not None:
            #clear the ax so if another file is uploaded (to be verified since m unsing the canvas draw method)
            self.parent.ax.plot(self.parent.ppm_data,self.parent.loaded_data,label='Spectra data',color='black')
            self.parent.canvas.draw()

    def load_fits(self):
        # file filters for Json files and Xml files
        file_filters = "spectra files (*.json *.xml)"

        # Prompt the user to select a file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filters)
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as file:
                            data = json.load(file)
                            for metabolite_data in data:
                                metabolite_name = metabolite_data.get('name')
                                if metabolite_name:
                                    #print("Width:", self.parent.width_values[metabolite_name])
                                    #print("Amplitude:", self.parent.amplitude_values[metabolite_name])
                                    #print("Type:", self.parent.type_values[metabolite_name])
                                     
                                    self.update_component_values(
                                metabolite_name,
                                metabolite_data.get('Width'),
                                metabolite_data.get('Amplitude'),
                                metabolite_data.get('Type')
                            )
                                self.parent.update_plot()
                if file_path.endswith('.xml'): 
                    with open(file_path, 'r') as file:
                            tree = ET.parse(file_path)
                            print(tree)
                            root = tree.getroot()
                            print(root)
                            
                            for metab in root.findall('Metabolite'):
                                print(metab)
                                metabolite_name = metab.get('name')
                                
                                #print("Width:", self.parent.width_values[metabolite_name])
                                #print("Amplitude:", self.parent.amplitude_values[metabolite_name])
                                #print("Type:", self.parent.type_values[metabolite_name]
                                self.update_component_values(
                                metabolite_name,
                                float(metab.find('Width').text),
                                float(metab.find('Amplitude').text),
                                metab.find('Type').text
                            )
                                self.parent.update_plot()
         
            except Exception as e:
                # Handle errors during the loading process
                print("Error loading file:", e)
                QMessageBox.information(self, "Information", "Fits file corrupted.")


  
    def update_component_values(self,metabolite_name, width, amplitude, type_):
                
                self.parent.largeur_mh[metabolite_name] = width
                self.parent.amplitude_values[metabolite_name] = amplitude
                self.parent.type_values[metabolite_name] = type_
                self.parent.metab_visibility_states[metabolite_name] = True

                # Update the control bar checkboxes and radio buttons
                self.parent.control_bar.metab_checkboxes[metabolite_name].setChecked(True)
                #print(type_)
                if type_ == "Gaussienne":
                    self.parent.control_bar.gaussien_radiobuttons[metabolite_name].setChecked(True)
                  
                else:
                    self.parent.control_bar.lorentzien_radiobuttons[metabolite_name].setChecked(True)

                # Set slider values
                #print(width)
                #print(amplitude)
                self.parent.control_bar.width_sliders[metabolite_name].setValue(int(width*1000))
                self.parent.control_bar.amplitude_sliders[metabolite_name].setValue(int(amplitude/100))