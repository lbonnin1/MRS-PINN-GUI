
import csv
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import numpy as np
import re


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

basisset='' #path to the basis set csv file
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ControlBar(QWidget):
    """Generates a controlbar for metabolites manipulation.
    The controlbar components are :
        Sliders: to control amplitude and width.
        Radio button: to control the metabolite's curve type.
        Checkbox: to control the metabolite's curve visibility.
    The controlbar methods are :
        create_metabolite_control: creates control section of a metabolite for properties adjustement.
        update_width: updates the width value of a metabolite's curve on the plot based on the slider value.
        update_amplitude: updates the amplitude value of a metabolite's curve on the plot based on the slider value.
        update_type: updates the type of a metabolite's curve on the plot based on the radiobutton state.
        metab_update_visibility: updates the visibility state of a metabolite's curve on the plot based on the checkerbox state.
    """

    def __init__(self, parent):
        """Initializes the ControlBar widget.

        Args:
            parent (QWidget): The parent widget to which this ControlBar belongs.

        Attributes:
            parent (QWidget): The parent widget.
            width_values (dict): Dictionary to store width values for each metabolite.
            amplitude_values (dict): Dictionary to store amplitude values for each metabolite.
            type_values (dict): Dictionary to store type values for each metabolite.
            metab_visibility_states (dict): Dictionary to store visibility states for each metabolite.
            control_layout (QVBoxLayout): Layout manager for the ControlBar widget.
            scroll_area (QScrollArea): Scroll area to enable scrolling for the control widget.
            control_widget (QWidget): Widget containing all control elements.
            control_widget_layout (QVBoxLayout): Layout manager for the control widget.
            group_metabolites (QGroupBox): Group box to contain controls for metabolites.
            layout_metabolites (QVBoxLayout): Layout manager for metabolite controls.

        """
        super(ControlBar, self).__init__(parent)
        # Store references to parent widget and initial values

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        self.parent = parent 
        self.largeur_mh = parent.largeur_mh
        self.amplitude_values = parent.amplitude_values
        self.type_values = parent.type_values
        self.metab_visibility_states = parent.metab_visibility_states
        self.basis_visibility_states = {metabolite: False for metabolite in self.parent.metab_pro}
        self.metab_checkboxes={}
        self.basis_checkboxes={}
        self.gaussien_radiobuttons={}
        self.lorentzien_radiobuttons={}
        self.width_sliders={}
        self.amplitude_sliders={}
        

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #uploading the basisset csv file and retrieving the first column values which represents the ppm range values
        with open(basisset, 'r') as file:
            reader = csv.reader(file,delimiter=';')
            parent.basisset_data= next(reader)
            for row in reader:
                parent.ppm_data.append(float(row[0])) 
            parent.ppm_data=np.array(parent.ppm_data)
     
        # Main layout for ControlBar
        self.control_layout = QVBoxLayout()   
        search_layout = QHBoxLayout()
        
        search_label = QLabel("Search Metabolite:")
        self.search_line_edit = QLineEdit()
        self.search_line_edit.textChanged.connect(self.filter_metabolites)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_line_edit)
      
        self.control_layout.addLayout(search_layout)
       
        # Scroll area to enable scrolling for the control widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) 

        # Widget containing all control elements
        self.control_widget = QWidget()
        self.control_widget_layout = QVBoxLayout(self.control_widget)
        self.checkboxgroup = QCheckBox('Select all')
        self.checkboxgroup.setChecked(False)
        self.checkboxgroup.stateChanged.connect(lambda state: self.all_update_visibility(state))
        self.control_widget_layout.addWidget(self.checkboxgroup)
      
        # Create controls for each metabolite
        for i, (metabolite,pro) in enumerate(zip(parent.metabolites,self.parent.metab_pro)):
            
            layout_box = QVBoxLayout()
            h_layout=QHBoxLayout()

            #update the curve's visbility
            self.checkboxvisibility = QCheckBox('Metabolite')
            self.checkboxvisibility.setChecked(False)
            self.checkboxvisibility.stateChanged.connect(lambda state, m=metabolite: self.metab_update_visibility(state, m))
            h_layout.addWidget(self.checkboxvisibility)
            group_box = QGroupBox(metabolite)
            color_name = list(self.parent.colors.keys())[i % len(self.parent.colors)]
            color_value = self.parent.colors[color_name]
            self.checkboxvisibility.color=color_value
            self.checkboxvisibility.setStyleSheet(
                 f"""
             
                QCheckBox::indicator:checked {{
                    background-color: rgb({color_value[0] // 2}, {color_value[1] // 2}, {color_value[2] // 2});
                }}
                """
            )
            self.metab_checkboxes[metabolite] = self.checkboxvisibility

            self.basisvisibility = QCheckBox('Basis set')
            self.basisvisibility.setChecked(False)
            self.basisvisibility.stateChanged.connect(lambda state, m =pro: self.basis_update_visibility(state, m))
            self.basis_checkboxes[ re.sub(r'\d+\s*$', '', metabolite).strip()]=self.basisvisibility
            h_layout.addWidget(self.basisvisibility)

            #update the curve's type
            radiobutton_L = QRadioButton("Lorentzienne")
            radiobutton_L.setChecked(True)
            radiobutton_L.type = "Lorentzienne"
            radiobutton_L.toggled.connect(lambda state, m=metabolite: self.update_type(state, m))
            h_layout.addWidget(radiobutton_L)
            self.lorentzien_radiobuttons[metabolite]=radiobutton_L
            radiobutton_G = QRadioButton("Gaussienne")
            radiobutton_G.type = "Gaussienne"
            radiobutton_G.toggled.connect(lambda state, m=metabolite: self.update_type(state, m))
            self.gaussien_radiobuttons[metabolite]=radiobutton_G
            h_layout.addWidget(radiobutton_G)
            color_style = f"""
                QRadioButton::indicator {{
                    width: 11px;
                    height: 11px;
                    border-radius: 6px;  /* Makes the indicator round */
                    border: 1px solid #555;  
                    background-color: #FFFFFF
                }}

                QRadioButton::indicator:checked {{
                    background-color: rgb({color_value[0] // 2}, {color_value[1] // 2}, {color_value[2] // 2});
                }}
                """

            radiobutton_L.setStyleSheet(color_style)
            radiobutton_G.setStyleSheet(color_style)

         
            #update the curve's width
            width_label = QLabel("Width:")
            layout_box.addWidget(width_label)
            slider_width = QSlider(Qt.Horizontal)
            slider_width.setMinimum(1)
            slider_width.setMaximum(300)
            slider_width.setValue(150)
            slider_width.setSingleStep(1)
            slider_width.valueChanged.connect(lambda value, m=metabolite: self.update_width(value, m))
            self.width_sliders[metabolite]=slider_width
            layout_box.addWidget(slider_width)
    
            #update the curve's amplitude
            amplitude_label = QLabel("Amplitude:")
            layout_box.addWidget(amplitude_label)
            slider_amplitude = QSlider(Qt.Horizontal)
            slider_amplitude.setMinimum(0)
            slider_amplitude.setMaximum(1500)
            slider_amplitude.setValue(750)
            slider_amplitude.setSingleStep(10)
             
            slider_amplitude.valueChanged.connect(lambda value, m=metabolite: self.update_amplitude(value, m))
            self.amplitude_sliders[metabolite]=slider_amplitude
            layout_box.addWidget(slider_amplitude)

       
            combinedLayout = QVBoxLayout()
            
            combinedLayout.addLayout(h_layout)
            combinedLayout.addLayout(layout_box)
            group_box.setLayout(combinedLayout)
            
            
            # Add group metabolites to the control widget layout
            self.control_widget_layout.addWidget(group_box)
            self.scroll_area.setWidget(self.control_widget)

            # Add scroll area to the main layout
            self.control_layout.addWidget(self.scroll_area)
            self.setLayout(self.control_layout)



    def filter_metabolites(self, text):
        """Filters metabolites based on the given text.

            Args:
                text (str): The text to filter metabolites.

            Returns:
                None
        """
        layout=self.layout_metabolites
        # Iterate over the items in the layout in reverse order
        for j in reversed(range(layout.count())):
            # Get the item at the current index in the layout
            widget_item = layout.itemAt(j)
            # Check if the item is not None and if it has a widget
            if widget_item is not None and widget_item.widget() is not None:
                # Get the widget from the item
                widget = widget_item.widget()
                # Print the object name of the widget
                print("Widget object name:", widget.objectName())
                # Check if the text is found in the object name of the widget
                if text.lower() in widget.objectName().lower():
                    # If a match is found, show the widget
                    print("Match found, showing widget")
                    widget.show()
                else:
                    # If no match is found, hide the widget
                    print("No match, hiding widget")
                    widget.hide()

    def update_width(self, value, metabolite):
        """Updates the width value of a metabolite's curve on the plot based on the slider value.
        Args:
            value (float): The width value returned by manipulating the slider .
            metabolite (str): The name of the metabolite being updated.

        Returns:
            None
        """
        self.largeur_mh[metabolite] = value / 1000
        self.parent.update_plot()

   
    def update_amplitude(self, value, metabolite):
        """Updates the amplitude value of a metabolite's curve on the plot based on the slider value.
        Args:
            value (float): The amplitude value returned by manipulating the slider .
            metabolite (str): The name of the metabolite being updated.

        Returns:
            None
        """
        self.amplitude_values[metabolite] = value*100
        self.parent.update_plot()
    
    
    def update_type(self, state, metabolite):
        """Updates the type of a metabolite's curve on the plot based on the radiobutton state.
        Args:
            state (bool): The state of the radiobutton indicating whether it is checked or not.
            metabolite (str): The name of the metabolite being updated.

        Returns:
            None
        """
        sender = self.sender()
        if state and isinstance(sender, QRadioButton):
            self.type_values[metabolite] = sender.text()
            self.parent.update_plot()

   
    def metab_update_visibility(self, state, metabolite):
        """Updates the visibility state of a metabolite's curve on the plot based on the checkerbox state.
        Args:
            state (bool): The state of the checkerbox indicating whether it is checked or not.
            metabolite (str): The name of the metabolite being updated.

        Returns:
            None
        """
        self.metab_visibility_states[metabolite] = state
        self.parent.update_plot()

    def all_update_visibility(self, state):
        for metab in  self.parent.metabolites:
            self.metab_visibility_states[metab] = state
            self.metab_checkboxes[metab].setChecked(state)
     
        self.parent.update_plot()
            

    def basis_update_visibility(self,state,metab):
        """Updates the visibility state of a metabolite's basis set reference curve on the plot based on the checkbox state.
        Args:
            state (bool): The state of the checkerbox indicating whether it is checked or not.
            metabolite (str): The name of the metabolite being updated.

        Returns:
            None
        """
        self.basis_visibility_states[metab] = state  

    
        self.parent.update_plot()
#-----------------------------------------------------------------------------Plot the basis set reference data-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    
    def plot_basisset(self):
        """
        Plot the basisset data for visible metabolites.

        Returns:
            None
        """
        
        for metab in self.parent.metab_pro:
            subs = {}
            if self.basis_visibility_states[metab]:
                print(f'checked' + metab)
                if '+' in metab:
                    subs = {sub.strip() for sub in metab.split('+')}
                    print(subs)
                else:
                    subs = {metab}

                for sub in subs:
                    y = np.empty_like(self.parent.ppm_data, dtype=np.complex128)  # Initialize empty array
                    
                    # Read basis set csv file
                    with open(basisset, 'r') as file:
                        reader = csv.reader(file, delimiter=';')
                        # Get header
                        header_row = next(reader)
                        print("here")
        
                        # Get the index of the column of the metabolite
                        metab_index = header_row.index(sub)
                        print(metab_index)
                        y_values = []
                        for row in reader:
                            y_value = complex(row[metab_index].replace("i", "j"))  # Convert string to complex number
                            y_values.append(y_value)  # Store data in a list
                        
                        # Convert the list to a numpy array
                        y = np.array(y_values, dtype=np.complex128)  
                        
                        # Plot the data
                        self.parent.ax.plot(self.parent.ppm_data, np.abs(y) * 1200, label=metab, color="grey")
           