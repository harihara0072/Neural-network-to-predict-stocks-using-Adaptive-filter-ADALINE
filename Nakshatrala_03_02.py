# Nakshatrala, Hari Hara Kumar
# 1001-102-740
# 2017-09-17
# Assignment_03_02

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
import scipy.misc
import os
import mpmath

class DisplayActivationFunctions:


    def __init__(self, root, master, *args, **kwargs):

        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 110
        self.xasis = np.arange(1000)
        self.learning_rate = 0.001



        self.learn_count = 0
        self.activation_function = "Symmetrical Hard limit"
        self.learning_method = 'Filtered Learning'
        self.files_imported = self.read_csv_as_matrix("stock_data.csv")
        self.number_of_delayed_elements = 10
        self.training_sample_size = 80
        self.batch_size = 100
        self.number_of_iterations = 10
        self.weight_matrix = 0
        self.total_files = np.shape(self.files_imported)[0]
        self.normal_vector = self.normalize_data_set()
        self.train_set = self.generate_train_set()
        self.test_set = self.generate_test_set()
        self.price_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements+1), dtype=np.float32)
        self.volume_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements+1), dtype=np.float32)
        self.price_sqaure_error = []
        self.price_absolute_error = []
        self.volume_square_error = []
        self.volume_absolute_error = []
        self.batch_count = 0
        self.batch_number = 0
        self.batch_array = []
        self.price_mse_array = []
        self.price_mae_array = []
        self.volume_mse_array = []
        self.volume_mae_array = []


        #########################################################################
        #  Set up the plotting area
        #########################################################################

        #citatiom: Matplotlib subplots examples(https://matplotlib.org/examples/pylab_examples/subplot_demo.html) are taken as reference to create subplots
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure()

        self.p1 = self.figure.add_subplot(2, 2, 1)
        self.p1.set_xlabel("Batch")
        self.p1.set_ylabel("MSE")
        self.p1.set_xlim(self.xmin, self.xmax)
        self.p1.set_ylim(self.ymin, self.ymax)

        self.p2 = self.figure.add_subplot(2, 2, 2)
        self.p2.set_xlim(self.xmin, self.xmax)
        self.p2.set_ylim(self.ymin, self.ymax)
        self.p2.set_xlabel("Batch ")
        self.p2.set_ylabel("MSE")

        self.p3 = self.figure.add_subplot(2, 2, 3)
        self.p3.set_xlim(self.xmin, self.xmax)
        self.p3.set_ylim(self.ymin, self.ymax)
        self.p3.set_xlabel("Batch")
        self.p3.set_ylabel("MAE")

        self.p4 = self.figure.add_subplot(2, 2, 4)
        self.p4.set_xlim(self.xmin, self.xmax)
        self.p4.set_ylim(self.ymin, self.ymax)
        self.p4.set_xlabel("Batch")
        self.p4.set_ylabel("MAE")

        self.p1.set_title("Price - MSE")
        self.p2.set_title("Price - MAE")
        self.p3.set_title("Volume - MSE ")
        self.p4.set_title("Volume - MAE ")
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

      
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        # setting learning rate slider
        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=0.1, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Learning Rate",
                                            command=lambda event: self.learning_rate_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_callback())
        self.learning_rate_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #slider for number of delayed elements
        self.number_of_delayed_elements_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Number of Delayed Elements",
                                             command=lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.set(self.number_of_delayed_elements)
        self.number_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #slider for Training sample size(Percentage)

        self.training_sample_size_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(),
                                                          orient=tk.HORIZONTAL,
                                                          from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                          activebackground="#FF0000",
                                                          highlightcolor="#00FFFF",
                                                          label="Training sample size(Percentage)",
                                                          command=lambda
                                                              event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.set(self.training_sample_size)
        self.training_sample_size_slider.bind("<ButtonRelease-1>",
                                                    lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)


        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(6, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')

        # slider Batch size

        self.batch_size_slider = tk.Scale(self.buttons_frame, variable=tk.DoubleVar(),
                                          orient=tk.HORIZONTAL,
                                          from_=1, to_=200, resolution=1, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Batch Size",
                                          command=lambda
                                              event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>",
                                    lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #slider for number of iterations

        self.number_of_iterations_slider = tk.Scale(self.buttons_frame, variable=tk.DoubleVar(),
                                          orient=tk.HORIZONTAL,
                                          from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Number of Iterations",
                                          command=lambda
                                              event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.set(self.number_of_iterations)
        self.number_of_iterations_slider.bind("<ButtonRelease-1>",
                                    lambda event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #button for making zero weights

        self.make_weights_to_zero_button = tk.Button(self.buttons_frame, text="Make Weights to Zero", command=self.make_weights_to_zero)
        self.make_weights_to_zero_button.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # button for Adjust weights

        self.adjust_weights_button = tk.Button(self.buttons_frame, text="Adjust Weights",
                                                     command=self.adjust_weights_button_callback)
        self.adjust_weights_button.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def net_value_callback(self, w, i):

        self.net_value = np.zeros((10, 200), dtype=np.float32)
        self.net_value[:] = np.dot(w, i)



        return self.net_value

        #self.display_activation_function()

    def plot_the_error(self,count):

        non_matching_coloumns = 0
        for i in range(0, 200):
            max_value = np.max(self.net_value[:, i])
            for j in range(0, 10):
                if self.net_value[j, i] == max_value:
                    self.net_value[j, i] = 1
                else:
                    self.net_value[j, i] = 0

        for t in range(0, 200):
            if np.array_equal(self.net_value[:, t], self.test_target_vector[:, t]) == True:
                non_matching_coloumns = non_matching_coloumns
            else:
                non_matching_coloumns = non_matching_coloumns + 1



        error_percentage = (non_matching_coloumns/2)
        #self.error_matrix[count] = error_percentage
        return error_percentage

    def display_activation_function(self):

       
        self.p1.set_xlim(self.xmin, len(self.batch_array))
        self.p1.set_ylim(0, 2)
        self.p2.set_xlim(self.xmin, len(self.batch_array))
        self.p2.set_ylim(0, 2)
        self.p3.set_xlim(self.xmin, len(self.batch_array))
        self.p3.set_ylim(0, 2)
        self.p4.set_xlim(self.xmin, len(self.batch_array))
        self.p4.set_ylim(0, 2)
        self.p1.set_xlabel("Batch")
        self.p1.set_ylabel("Error")
        self.p2.set_xlabel("Batch ")
        self.p2.set_ylabel("Error")
        self.p3.set_xlabel("Batch")
        self.p3.set_ylabel("Error")
        self.p4.set_xlabel("Batch")
        self.p4.set_ylabel("Error")

        self.p1.set_title("Price - Mean Square Error")
        self.p2.set_title("Price - Maximum Absolute Error ")
        self.p3.set_title("Volume - Maximum Square Error ")
        self.p4.set_title("Volume - Maximum Absolute Error ")


        self.p1.xaxis.set_visible(True)
        self.p1.yaxis.set_visible(True)
        self.p2.xaxis.set_visible(True)
        self.p2.yaxis.set_visible(True)
        self.p3.xaxis.set_visible(True)
        self.p3.yaxis.set_visible(True)
        self.p4.xaxis.set_visible(True)
        self.p4.yaxis.set_visible(True)
        #plt.title(self.learning_method + " " +self.activation_function)
        self.canvas.draw()

    def learning_rate_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def number_of_delayed_elements_slider_callback(self):
        self.number_of_delayed_elements = self.number_of_delayed_elements_slider.get()
        self.price_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements))
        self.volume_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements))

    def training_sample_size_slider_callback(self):
        self.training_sample_size = self.training_sample_size_slider.get()
        x = self.get_train_set_number()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()

    def make_weights_to_zero(self):
        self.price_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements), dtype=np.float32)
        self.volume_weight_matrix = np.zeros((1, 2 * self.number_of_delayed_elements), dtype=np.float32)
        #self.axes.cla()
        #self.axes.cla()
        self.price_sqaure_error = []
        self.price_absolute_error = []
        self.volume_square_error = []
        self.volume_absolute_error = []
        self.batch_count = 0
        self.batch_number = 0
        self.batch_array = []
        self.price_mse_array = []
        self.price_mae_array = []
        self.volume_mse_array = []
        self.volume_mae_array = []
        self.p1.cla()
        self.p2.cla()
        self.p3.cla()
        self.p4.cla()
        self.canvas.draw()

    def adjust_weights_button_callback(self):

        self.train_set = self.generate_train_set()
        self.test_set = self.generate_test_set()
        train_set_count =self.get_train_set_number()
        test_set_count = self.total_files - train_set_count

        for a in range (0, self.number_of_iterations):
            for i in range(self.number_of_delayed_elements, int(train_set_count)):
                self.batch_count = self.batch_count + 1
                input = self.train_set[i - self.number_of_delayed_elements: i, :]
                input = np.reshape(input, (2*self.number_of_delayed_elements))
                np.insert(input,2*self.number_of_delayed_elements,1)
                #print("input is ")
                #print(input)

                self.price_net_value = np.dot(self.price_weight_matrix, input)
                self.volume_net_value = np.dot(self.volume_weight_matrix, input)
                self.target_matrix = self.train_set[i, :]
                price_error = self.target_matrix[0] - self.price_net_value
                volume_error = self.target_matrix[1] - self.volume_net_value

                self.price_weight_matrix = self.price_weight_matrix + (2 * self.learning_rate * price_error * input)
                self.volume_weight_matrix = self.volume_weight_matrix + (2 * self.learning_rate * volume_error * input)
                #print("price_weights")
                #print(self.price_weight_matrix)
                if (self.batch_count % self.batch_size) == 0:
                    self.batch_number = self.batch_number + 1
                    self.batch_array.append(self.batch_number)
                    for j in range(self.number_of_delayed_elements, int(test_set_count)):
                        test_input = self.test_set[j - self.number_of_delayed_elements: j, :]
                        test_input = np.reshape(test_input, (2*self.number_of_delayed_elements))
                        self.test_price_output = np.dot(self.price_weight_matrix, test_input)
                        self.test_volume_output = np.dot(self.volume_weight_matrix, test_input)
                        test_target = self.test_set[j, :]
                        test_price_error = (test_target[0] - self.test_price_output)
                        test_volume_error = (test_target[1] - self.test_volume_output)

                        self.price_absolute_error.append(np.abs(test_price_error[0]))
                        self.price_sqaure_error.append(np.abs(test_price_error[0]*test_price_error[0]))
                        self.volume_absolute_error.append(np.abs(test_volume_error[0]))
                        self.volume_square_error.append(np.abs(test_volume_error[0]*test_volume_error[0]))
                    price_mse = self.calculate_the_mean(self.price_sqaure_error)/len(self.price_sqaure_error)
                    price_mae = self.calculate_the_mean(self.price_absolute_error)/len(self.price_absolute_error)
                    volume_mse = self.calculate_the_mean(self.volume_square_error)/len(self.volume_square_error)
                    volume_mae = self.calculate_the_mean(self.volume_absolute_error)/len(self.volume_absolute_error)

                    self.price_mse_array.append(price_mse)
                    self.price_mae_array.append(price_mae)
                    self.volume_mae_array.append(volume_mae)
                    self.volume_mse_array.append(volume_mse)
                else:
                    continue

        self.p1.plot(self.batch_array, self.price_mse_array, color='blue')
        self.p2.plot(self.batch_array,self.price_mae_array, color='r')
        self.p3.plot(self.batch_array, self.volume_mse_array, color='k')
        self.p4.plot(self.batch_array, self.volume_mae_array, color='m')

        self.display_activation_function()
        self.canvas.draw()

    def calculate_the_mean(self,function):
        sum = 0
        for i in range(0, len(function)):
            sum = sum + function[i]
        return sum

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        #self.display_activation_function()

    def number_of_iterations_slider_callback(self):
        self.number_of_iterations = self.number_of_iterations_slider.get()

    def learning_method_dropdown_callback(self):
        self.learning_method = self.Learning_method_variable.get()
        #self.display_activation_function()

    def read_csv_as_matrix(self, file_name):
        # Each row of data in the file becomes a row in the matrix
        # So the resulting matrix has dimension [num_samples x sample_dimension]
        data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
        return data

    '''def generate_train_set(self):
        self.train_set = np.zeros((785, 800))
        for i in range(0, 800):
            self.train_set[0:784, i] = self.read_one_image_and_convert_to_vector(self.learn_file_names[i])
        self.train_set[784, :] = 1
        self.train_set = (self.train_set/255) - 0.5
        return self.train_set'''

    def generate_train_set(self):
        a = self.normal_vector
        size = self.get_train_set_number()
        train_matrix = a[0:int(size), :]
        return train_matrix

    def generate_test_set(self):
        a = self.normal_vector
        size = self.get_train_set_number()
        test_matrix = a[int(size):self.total_files, :]
        return test_matrix

    def get_train_target(self):
        self.train_target = np.zeros((10, 800))
        for i in range(0, 800):
            str = self.learn_file_names[i]
            if str[0] == "0":
                self.train_target[0, i] = 1
            elif str[0] == "1":
                self.train_target[1, i] = 1
            elif str[0] == "2":
                self.train_target[2, i] = 1
            elif str[0] == "3":
                self.train_target[3, i] = 1
            elif str[0] == "4":
                self.train_target[4, i] = 1
            elif str[0] == "5":
                self.train_target[5, i] = 1
            elif str[0] == "6":
                self.train_target[6, i] = 1
            elif str[0] == "7":
                self.train_target[7, i] = 1
            elif str[0] == "8":
                self.train_target[8, i] = 1
            elif str[0] == "9":
                self.train_target[9, i] = 1
        return self.train_target

    '''def caluculate_the_error(self):
        train_net_value = np.zeros((10,800), dtype=np.float32)
        train_net_value[:] = np.dot(self.weight_matrix, self.train_set)
        if self.activation_function == 'Symmetrical Hard limit':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] > 0:
                        self.train_output_vector[i, j] = 1
                    else:
                        self.train_output_vector[i, j] = -1

        elif self.activation_function == 'Linear':
            for i in range(0, 10):
                for j in range(0, 800):
                    if train_net_value[i, j] < 1000:
                        self.train_output_vector[i, j] = np.float32(train_net_value[i, j])
                    else:
                        self.train_output_vector[i, j] = 1000
                        # o[i, j] = activation

        elif self.activation_function == 'Hyperbolic Tangent':
            for i in range(0, 10):
                for j in range(0, 800):
                    train_net_value[i, j] = train_net_value[i, j ]/1000000
                    activation = np.round((np.exp(train_net_value[i, j]) - np.exp(-train_net_value[i, j])) / (
                    np.exp(train_net_value[i, j]) + np.exp(-train_net_value[i, j])))
                    self.train_output_vector[i, j] = activation

        error = self.train_target_vector - self.train_output_vector

        return error'''

    def get_test_target(self):
        self.test_target = np.zeros((10, 200))
        for i in range(0, 200):
            str = self.test_file_names[i]
            if str[0] == "0":
                self.test_target[0, i] = 1
            elif str[0] == "1":
                self.test_target[1, i] = 1
            elif str[0] == "2":
                self.test_target[2, i] = 1
            elif str[0] == "3":
                self.test_target[3, i] = 1
            elif str[0] == "4":
                self.test_target[4, i] = 1
            elif str[0] == "5":
                self.test_target[5, i] = 1
            elif str[0] == "6":
                self.test_target[6, i] = 1
            elif str[0] == "7":
                self.test_target[7, i] = 1
            elif str[0] == "8":
                self.test_target[8, i] = 1
            elif str[0] == "9":
                self.test_target[9, i] = 1
        return self.test_target

    def get_random_weights(self):
        self.weight_matrix = np.round([np.random.uniform(-0.001, 0.001, [1, 785])], decimals=6)
        self.learn_count = 0
        self.axes.cla()
        self.axes.cla()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error Percentage')
        self.axes.set_title(self.learning_method +" "+ self.activation_function)
        self.canvas.draw()

    def test_the_weights(self):
        self.net_value_callback(w=self.weight_matrix,i=self.test_set)

    def get_train_set_number(self):
        train_set_number = np.round((self.training_sample_size / 100) * self.total_files)
        return train_set_number

    def normalize_data_set(self):
        price_max = np.amax(self.files_imported[:, 0])
        volume_max = np.amax(self.files_imported[:, 1])

        for i in range(0, self.total_files):
            self.files_imported[i, 0] = (self.files_imported[i, 0]/price_max) - 0.5
            self.files_imported[i, 1] = (self.files_imported[i, 1]/volume_max) - 0.5

        return self.files_imported


















