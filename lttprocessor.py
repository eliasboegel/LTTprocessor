import os, timeit, operator
import numpy
from matplotlib import pyplot, transforms, ticker
from PIL import Image, ImageDraw

def generate_plot_thermal(dir, outputdir, prefix):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for root, dirs, files in os.walk(dir):
        print(f"Processing {root}")
        
        # Declare temperature array
        temperature_array = 0

        # Keep count of how many files have been treated, also needed to create array and add subsequent readings to first array
        filecounter = 1

        

        for file in files:
            # If file is csv file, then use it
            test = file.rsplit(".", 1)
            if (test[-1] == "csv"):
                path = os.path.join(dir, file)

                # If file is first file that is read
                if not (filecounter > 1):
                    # Read the csv file and create array from it
                    temperature_array = numpy.genfromtxt(os.path.join(root, file), dtype = numpy.float, delimiter = ";")
                else:
                    # Read the csv file and add array results to previous ones
                    add = numpy.genfromtxt(os.path.join(root, file), dtype = numpy.float, delimiter = ";")
                    temperature_array = temperature_array + add

                filecounter += 1

        if filecounter > 1:
            # Average temperatures
            temperature_array = temperature_array / filecounter

            T_max = numpy.nanmax(temperature_array)
            T_min = numpy.nanmin(temperature_array)
            T_range = T_max - T_min

            # Create mask to cut out wing
            wingshape = [(209,0),(485,0),(470,472),(404,476),(470,472),(324,472),(199,469)]
            x_min = min(wingshape)[0]
            x_max = max(wingshape)[0]
            y_min = min(wingshape, key=operator.itemgetter(1))[1]
            y_max = max(wingshape, key=operator.itemgetter(1))[1]

            img_wingmask = Image.new('L', (temperature_array.shape[1], temperature_array.shape[0]), 0)
            ImageDraw.Draw(img_wingmask).polygon(wingshape, outline=1, fill=1)
            wingmask = numpy.array(img_wingmask)


            masked_temperature_array = numpy.ma.array(temperature_array, mask = numpy.logical_not(wingmask))
            trimmed_masked_temperature_array = numpy.fliplr(masked_temperature_array[y_min:y_max, x_min:x_max])


            pyplot.imshow(trimmed_masked_temperature_array, cmap = 'jet')
            cb = pyplot.colorbar()
            cb.set_label(r"T [°C]", rotation = 0, labelpad=30)
            ax = pyplot.gca()
        
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(x_max - x_min))
            ax.xaxis.set_major_locator(ticker.LinearLocator(5))
            ax.set_xlim(left = 0)
            ax.set_xlabel(r"c [-]")
        
            pyplot.savefig(os.path.join(outputdir, f"{prefix}_{os.path.basename(os.path.normpath(root))}_highrange.png"), transparent = True, dpi = 300, bbox_inches='tight')
            pyplot.clf()

            pyplot.imshow(trimmed_masked_temperature_array, cmap = 'jet', vmin = T_min + 0.7 * T_range, vmax = T_max)
            cb = pyplot.colorbar()
            cb.set_label(r"T [°C]", rotation = 0, labelpad=30)
            ax = pyplot.gca()
        
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(x_max - x_min))
            ax.xaxis.set_major_locator(ticker.LinearLocator(5))
            ax.set_xlim(left = 0)
            ax.set_xlabel(r"c [-]")
            
            pyplot.savefig(os.path.join(outputdir, f"{prefix}_{os.path.basename(os.path.normpath(root))}_lowrange.png"), transparent = True, dpi = 300, bbox_inches='tight')


            pyplot.clf()





def generate_data_exp_cp(path):
    print(f"Processing {path}")

    data = numpy.genfromtxt(path, dtype = numpy.dtype(str), delimiter = "\t").T

    # Add last column that contains 1 if data point is a hysteresis data point and 0 if it isn't
    output_arr = numpy.empty_like(data, shape = (data.shape[0], data.shape[1] + 1))
    output_arr[:,:-1] = data

    highest_aoa = float(output_arr[1,1])
    for col in range(1, output_arr.shape[0]):
        if (float(output_arr[col, 1]) >= highest_aoa):
            highest_aoa = float(output_arr[col, 1])
            output_arr[col, -1] = str(0)
        else:
            output_arr[col, -1] = str(1)

    return output_arr

def generate_data_num_cp(path):
    print(f"Processing {path}")

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()

    filecounter = 0
    output_arr = 0
    for file in files:
        filepath = os.path.join(path, file)
        if "3D" in path:
            data_19 = numpy.genfromtxt(filepath, dtype = numpy.dtype(str), skip_header = 551, skip_footer = 513).T
            data_20 = numpy.genfromtxt(filepath, dtype = numpy.dtype(str), skip_header = 578, skip_footer = 486).T
            data_midspan = (data_19.astype(numpy.float) + data_20.astype(numpy.float)) * 0.5
            # data_midspan[1] = data_midspan[1] / 0.24
            


            if not filecounter:
                output_arr = numpy.empty_like(data_19, shape = (len(files) + 1, data_19.shape[1] + 1))
                AoA = file.split("_v", 1)[0].rsplit("a=", 1)[1]
                output_arr[0] = numpy.concatenate([numpy.array([AoA]), data_midspan[1].astype(numpy.dtype(str))])

            AoA = file.split("_v", 1)[0].rsplit("a=", 1)[1]
            output_arr[filecounter + 1] = numpy.concatenate([numpy.array([AoA]), data_midspan[-1].astype(numpy.dtype(str))])

        else: # 2D case
            data = numpy.genfromtxt(filepath, dtype = numpy.dtype(str), skip_header = 5).T

            if not filecounter:
                output_arr = numpy.empty_like(data, shape = (len(files) + 1, data.shape[1] + 1))
                AoA = file.split(")", 1)[0].rsplit("(", 1)[1]
                output_arr[0] = numpy.concatenate([numpy.array([AoA]), data[0]])

            AoA = file.split(")", 1)[0].rsplit("(", 1)[1]
            output_arr[filecounter + 1] = numpy.concatenate([numpy.array([AoA]), data[1]])

        filecounter += 1
        
    return output_arr

def generate_plot_cp(data, outputdir, prefix):
    # Plotting example of a pressure distribution (cp file)
    # Reynolds number can be accessed with data[dataset, 0]
    # Angle of Attack can be accessed with data[dataset, 1]

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Check if data is numerical or experimental to get correct file names later
    hysteresis_prefix = ""
    is_exp = 0
    if (data[0,0].strip() == "Re"):
        is_exp = 1


    # Produce graph of all datasets in the file
    for dataset in range(1, data.shape[0]):
        
        if (is_exp):
            if(int(data[dataset,-1])):
                hysteresis_prefix = "h"

        ax = pyplot.gca()
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.xaxis.set_major_locator(ticker.LinearLocator(5))
        ax.set_xlabel(r"c [-]")
        ax.set_ylabel(r"$C_p$ [-]", rotation = 0)
        ax.invert_yaxis()

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')


        pyplot.plot(data[0, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe") / data[0, 2].astype(numpy.float, casting = "unsafe"), data[dataset, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe"), marker = ".")
        pyplot.grid()
        

        #pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
        ax.set_xlim(left = 0, right = 1)
        #x_max = max(data[0, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe") / data[0, 2].astype(numpy.float, casting = "unsafe"))
        #x_min = min(data[0, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe") / data[0, 2].astype(numpy.float, casting = "unsafe"))
        #y_max = max(data[dataset, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe"))
        #y_min = min(data[dataset, 2:data.shape[1]-is_exp].astype(numpy.float, casting = "unsafe"))
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.xaxis.set_label_coords(x_max + 0.15 * (x_max - x_min), 0, transform = ax.transData)
        ax.yaxis.set_label_coords(0, y_max + 0.1 * (y_max - y_min), transform = ax.transData)

        for xlabel in ax.get_xticklabels():
            xlabel.set_horizontalalignment("left")
        for ylabel in ax.get_yticklabels():
            ylabel.set_verticalalignment("bottom")

        pyplot.savefig(os.path.join(outputdir, f"{prefix}_AoA{hysteresis_prefix}{round(float(data[dataset, is_exp].strip()),1)}.png"), transparent = True, dpi = 300, bbox_inches='tight')
        pyplot.clf()

def generate_plot_comp_cp(data_exp, data_num, outputdir, prefix):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    num_data_label = r"VPM Prediction"
    if "3D" in prefix:
        num_data_label = r"VLM Prediction"

    # Produce graph of all datasets in the file
    for dataset in range(1, data_num.shape[0]):
        row_slice = numpy.round(data_exp[1:,1].astype(numpy.float, casting = "unsafe"), 1)
        index_col = numpy.where(round(float(data_num[dataset, 0]), 1) == row_slice)
        if (index_col[0].size == 0):
            continue

        dataset_exp = index_col[0][0] + 1
       
        
        ax = pyplot.gca()
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.xaxis.set_major_locator(ticker.LinearLocator(5))
        ax.set_xlabel(r"c [-]")
        ax.set_ylabel(r"$C_p$ [-]", rotation = 0)
        ax.invert_yaxis()

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')


        exp_plot, = pyplot.plot(data_exp[0, 2:-1].astype(numpy.float, casting = "unsafe") / data_exp[0, 2].astype(numpy.float, casting = "unsafe"), data_exp[dataset_exp, 2:-1].astype(numpy.float, casting = "unsafe"), label = "Experimental data", marker = ".")
        num_plot, = pyplot.plot(data_num[0, 2:].astype(numpy.float, casting = "unsafe") / data_num[0, 2].astype(numpy.float, casting = "unsafe"), data_num[dataset, 2:].astype(numpy.float, casting = "unsafe"), label = num_data_label, marker = "D", markersize = 4)

        pyplot.legend(handles=[exp_plot, num_plot], bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.0)
        pyplot.grid()

        #pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
        ax.set_xlim(left = 0, right = 1)
        #x_max = max(max(data_exp[0, 2:-1].astype(numpy.float, casting = "unsafe")), max(data_num[0, 2:].astype(numpy.float, casting = "unsafe"))) / max(data_exp[0, 2].astype(numpy.float, casting = "unsafe"), data_num[0, 2].astype(numpy.float, casting = "unsafe"))
        #x_min = min(min(data_exp[0, 2:-1].astype(numpy.float, casting = "unsafe")), min(data_num[0, 2:].astype(numpy.float, casting = "unsafe"))) / min(data_exp[0, 2].astype(numpy.float, casting = "unsafe"), data_num[0, 2].astype(numpy.float, casting = "unsafe"))
        #y_max = max(max(data_exp[dataset_exp, 2:-1].astype(numpy.float, casting = "unsafe")), max(data_num[dataset, 2:].astype(numpy.float, casting = "unsafe")))
        #y_min = min(min(data_exp[dataset_exp, 2:-1].astype(numpy.float, casting = "unsafe")), min(data_num[dataset, 2:].astype(numpy.float, casting = "unsafe")))
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.xaxis.set_label_coords(x_max + 0.15 * (x_max - x_min), 0, transform = ax.transData)
        ax.yaxis.set_label_coords(0, y_max + 0.1 * (y_max - y_min), transform = ax.transData)

        for xlabel in ax.get_xticklabels():
            xlabel.set_horizontalalignment("left")
        for ylabel in ax.get_yticklabels():
            ylabel.set_verticalalignment("bottom")

        
        pyplot.savefig(os.path.join(outputdir, f"{prefix}_AoA{round(float(data_num[dataset, 0].strip()),1)}.png"), transparent = True, dpi = 300, bbox_inches='tight')
        pyplot.clf()





def generate_data_exp_coeff(path):
    print(f"Processing {path}")

    data = numpy.genfromtxt(path, dtype = numpy.dtype(str), delimiter = "\t", skip_header = 2).T

    output_arr = numpy.zeros((5, data.shape[1]))
    output_arr[0] = data[1].astype(numpy.float, casting = "unsafe") # AoA
    output_arr[1] = data[3].astype(numpy.float, casting = "unsafe") # CL
    output_arr[2] = data[2].astype(numpy.float, casting = "unsafe") # CD
    output_arr[3] = data[4].astype(numpy.float, casting = "unsafe") # CM

    # Add last column that contains 1 if data point is a hysteresis data point and 0 if it isn't
    highest_aoa = float(output_arr[0,0])
    for row in range(1, output_arr.shape[1]):
        if (float(output_arr[0, row]) >= highest_aoa):
            highest_aoa = float(output_arr[0, row])
            output_arr[4, row] = 0
        else:
            output_arr[4, row] = 1

    return output_arr

def generate_data_num_coeff(path):
    print(f"Processing {path}")

    data = numpy.genfromtxt(path, dtype = numpy.dtype(str), delimiter = "  ", skip_header = 11).T

    output_arr = numpy.zeros((4, data.shape[1]))
    output_arr[0] = data[0].astype(numpy.float, casting = "unsafe") # AoA
    output_arr[1] = data[1].astype(numpy.float, casting = "unsafe") # CL
    output_arr[2] = data[2].astype(numpy.float, casting = "unsafe") # CD
    output_arr[3] = data[4].astype(numpy.float, casting = "unsafe") # CM

    return output_arr

def generate_plot_coeff(data, outputdir, prefix):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # Format: (x_column, y_column)
    plot_columns = [(2,1), (0,1), (0,2), (0,3)]
    column_names = ["alpha", "Cl", "Cd", "Cm"]
    column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_l$ [-]", r"$C_d$ [-]", r"$C_m$ [-]"]
    if "3D" in prefix:
        column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_L$ [-]", r"$C_D$ [-]", r"$C_M$ [-]"]

    is_exp = 0
    if (data.shape[0] > 4):
        is_exp = 1

    for column_combo in plot_columns:
        ax = pyplot.gca()
        pyplot.xlabel(column_symbols[column_combo[0]], rotation = 0)
        pyplot.ylabel(column_symbols[column_combo[1]], rotation = 0)
        ax.spines['left'].set_position("zero")
        ax.spines['right'].set_color("none")
        ax.spines['bottom'].set_position("zero")
        ax.spines['top'].set_color("none")

        
        if (is_exp):
            data_split = numpy.hsplit(data, numpy.where(numpy.diff(data[4]))[0])
            data_norm = numpy.hsplit(data, numpy.where(numpy.diff(data[4]))[0] + 1)[0]
            data_hysteresis = numpy.hsplit(data, numpy.where(numpy.diff(data[4]))[0])[1]

            norm_plot, = pyplot.plot(data_norm[column_combo[0],:], data_norm[column_combo[1],:], label = r"increasing $\alpha$", marker = ".")
            hysteresis_plot, = pyplot.plot(data_hysteresis[column_combo[0],:], data_hysteresis[column_combo[1],:], label = r"decreasing $\alpha$", marker = "^", markersize = 4)
            pyplot.legend(handles=[norm_plot, hysteresis_plot], bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.0)
        else:
            pyplot.plot(data[column_combo[0],:], data[column_combo[1],:], marker = "D", markersize = 4)

        pyplot.grid()

        #pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
        #x_max = max(data[column_combo[0],:])
        #x_min = min(data[column_combo[0],:])
        #y_max = max(data[column_combo[1],:])
        #y_min = min(data[column_combo[1],:])
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()


        ax.xaxis.set_label_coords(x_max + 0.15 * (x_max - x_min), 0, transform = ax.transData)
        ax.yaxis.set_label_coords(0, y_max + 0.1 * (y_max - y_min), transform = ax.transData)

        for xlabel in ax.get_xticklabels():
            xlabel.set_horizontalalignment("left")
        for ylabel in ax.get_yticklabels():
            ylabel.set_verticalalignment("bottom")

        pyplot.savefig(os.path.join(outputdir, f"{prefix}_{column_names[column_combo[1]]}-{column_names[column_combo[0]]}.png"), transparent = True, dpi = 300, bbox_inches='tight')
        pyplot.clf()

def generate_plot_comp_coeff(data_exp, data_num, outputdir, prefix):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    num_data_label = r"VPM Prediction"
    if "3D" in prefix:
        num_data_label = r"VLM Prediction"

    # Format: (x_column, y_column)
    plot_columns = [(2,1), (0,1), (0,2), (0,3)]
    column_names = ["alpha", "Cl", "Cd", "Cm"]
    column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_l$ [-]", r"$C_d$ [-]", r"$C_m$ [-]"]
    if "3D" in prefix:
        column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_L$ [-]", r"$C_D$ [-]", r"$C_M$ [-]"]

    for column_combo in plot_columns:
        ax = pyplot.gca()
        pyplot.xlabel(column_symbols[column_combo[0]], rotation = 0)
        pyplot.ylabel(column_symbols[column_combo[1]], rotation = 0)
        ax.spines['left'].set_position("zero")
        ax.spines['right'].set_color("none")
        ax.spines['bottom'].set_position("zero")
        ax.spines['top'].set_color("none")


        data_exp_split = numpy.hsplit(data_exp, numpy.where(numpy.diff(data_exp[4]))[0])
        data_exp_norm = numpy.hsplit(data_exp, numpy.where(numpy.diff(data_exp[4]))[0] + 1)[0]
        data_exp_hysteresis = numpy.hsplit(data_exp, numpy.where(numpy.diff(data_exp[4]))[0])[1]

        exp_plot_norm, = pyplot.plot(data_exp_norm[column_combo[0],:], data_exp_norm[column_combo[1],:], label = r"Experimental data (increasing $\alpha$)", marker = ".")
        exp_plot_hysteresis, = pyplot.plot(data_exp_hysteresis[column_combo[0],:], data_exp_hysteresis[column_combo[1],:], label = r"Experimental data (decreasing $\alpha$)", marker = "^", markersize = 4)

        num_plot, = pyplot.plot(data_num[column_combo[0]], data_num[column_combo[1]], label = num_data_label, marker = "D", markersize = 4)

        pyplot.legend(handles=[exp_plot_norm, exp_plot_hysteresis, num_plot], bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.0)
        pyplot.grid()

        #x_max = max(max(data_exp[column_combo[0]]), max(data_num[column_combo[0]]))
        #x_min = min(min(data_exp[column_combo[0]]), min(data_num[column_combo[0]]))
        #y_max = max(max(data_exp[column_combo[1]]), max(data_num[column_combo[1]]))
        #y_min = min(min(data_exp[column_combo[1]]), min(data_num[column_combo[1]]))
        #pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()


        ax.xaxis.set_label_coords(x_max + 0.15 * (x_max - x_min), 0, transform = ax.transData)
        ax.yaxis.set_label_coords(0, y_max + 0.1 * (y_max - y_min), transform = ax.transData)

        for xlabel in ax.get_xticklabels():
            xlabel.set_horizontalalignment("left")
        for ylabel in ax.get_yticklabels():
            ylabel.set_verticalalignment("bottom")

        pyplot.savefig(os.path.join(outputdir, f"{prefix}_{column_names[column_combo[1]]}-{column_names[column_combo[0]]}.png"), transparent = True, dpi = 300, bbox_inches='tight')
        pyplot.clf()





def generate_plot_comp_2d3d_coeff(data_2d_exp, data_2d_num, data_3d_exp, data_3d_num, outputdir, prefix):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Format: (x_column, y_column)
    plot_columns = [(2,1), (0,1), (0,2), (0,3)]
    column_names = ["alpha", "Cl", "Cd", "Cm"]
    #column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_l$ [-]", r"$C_d$ [-]", r"$C_m$ [-]"]
    column_symbols = [r"$\alpha$ [$^\circ$]", r"$C_L$ [-]", r"$C_D$ [-]", r"$C_M$ [-]"]
        

    for column_combo in plot_columns:
        ax = pyplot.gca()
        pyplot.xlabel(column_symbols[column_combo[0]], rotation = 0)
        pyplot.ylabel(column_symbols[column_combo[1]], rotation = 0)
        ax.spines['left'].set_position("zero")
        ax.spines['right'].set_color("none")
        ax.spines['bottom'].set_position("zero")
        ax.spines['top'].set_color("none")


        #data_2d_exp_split = numpy.hsplit(data_2d_exp, numpy.where(numpy.diff(data_2d_exp[4]))[0])
        #data_2d_exp_norm = numpy.hsplit(data_2d_exp, numpy.where(numpy.diff(data_2d_exp[4]))[0] + 1)[0]
        #data_2d_exp_hysteresis = numpy.hsplit(data_2d_exp, numpy.where(numpy.diff(data_2d_exp[4]))[0])[1]

        #data_3d_exp_split = numpy.hsplit(data_3d_exp, numpy.where(numpy.diff(data_3d_exp[4]))[0])
        #data_3d_exp_norm = numpy.hsplit(data_3d_exp, numpy.where(numpy.diff(data_3d_exp[4]))[0] + 1)[0]
        #data_3d_exp_hysteresis = numpy.hsplit(data_3d_exp, numpy.where(numpy.diff(data_3d_exp[4]))[0])[1]


        #exp_plot_norm_2d, = pyplot.plot(data_2d_exp_norm[column_combo[0],:], data_2d_exp_norm[column_combo[1],:], label = r"Experimental 2D data (increasing $\alpha$)", marker = ".")
        exp_plot_norm_2d, = pyplot.plot(data_2d_exp[column_combo[0],:], data_2d_exp[column_combo[1],:], label = r"Experimental 2D data)", marker = ".")
        #exp_plot_hysteresis_2d, = pyplot.plot(data_2d_exp_hysteresis[column_combo[0],:], data_2d_exp_hysteresis[column_combo[1],:], label = r"Experimental 2D data (decreasing $\alpha$)", marker = "^", markersize = 4)
        num_plot_2d, = pyplot.plot(data_2d_num[column_combo[0]], data_2d_num[column_combo[1]], label = r"VPM Prediction", marker = "D", markersize = 4)

        #exp_plot_norm_3d, = pyplot.plot(data_3d_exp_norm[column_combo[0],:], data_3d_exp_norm[column_combo[1],:], label = r"Experimental 3D data (increasing $\alpha$)", marker = ".")
        exp_plot_norm_3d, = pyplot.plot(data_3d_exp[column_combo[0],:], data_3d_exp[column_combo[1],:], label = r"Experimental 3D data", marker = ".")
        #exp_plot_hysteresis_3d, = pyplot.plot(data_3d_exp_hysteresis[column_combo[0],:], data_3d_exp_hysteresis[column_combo[1],:], label = r"Experimental 3D data (decreasing $\alpha$)", marker = "^", markersize = 4)
        num_plot_3d, = pyplot.plot(data_3d_num[column_combo[0]], data_3d_num[column_combo[1]], label = r"VLM Prediction", marker = "D", markersize = 4)

        #pyplot.legend(handles=[exp_plot_norm_2d, exp_plot_hysteresis_2d, num_plot_2d, exp_plot_norm_3d, exp_plot_hysteresis_3d, num_plot_3d], bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.0)
        pyplot.legend(handles=[exp_plot_norm_2d, num_plot_2d, exp_plot_norm_3d, num_plot_3d], bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.0)
        pyplot.grid()

        #x_max = max(max(data_exp[column_combo[0]]), max(data_num[column_combo[0]]))
        #x_min = min(min(data_exp[column_combo[0]]), min(data_num[column_combo[0]]))
        #y_max = max(max(data_exp[column_combo[1]]), max(data_num[column_combo[1]]))
        #y_min = min(min(data_exp[column_combo[1]]), min(data_num[column_combo[1]]))
        #pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()


        ax.xaxis.set_label_coords(x_max + 0.15 * (x_max - x_min), 0, transform = ax.transData)
        ax.yaxis.set_label_coords(0, y_max + 0.1 * (y_max - y_min), transform = ax.transData)

        for xlabel in ax.get_xticklabels():
            xlabel.set_horizontalalignment("left")
        for ylabel in ax.get_yticklabels():
            ylabel.set_verticalalignment("bottom")

        pyplot.savefig(os.path.join(outputdir, f"{prefix}_{column_names[column_combo[1]]}-{column_names[column_combo[0]]}.png"), transparent = True, dpi = 300, bbox_inches='tight')
        pyplot.clf()






windtunnel_folder = "D:\courses\windtunnel\AE2130-II G25"

start = timeit.default_timer()

# Generate 2D plots
generate_plot_thermal(os.path.join(windtunnel_folder, "2D", "experimental", "thermal"), os.path.join(windtunnel_folder, "plots", "2D", "experimental", "thermal"), "2D_thermal")

data_exp_cp_2d = generate_data_exp_cp(os.path.join(windtunnel_folder, "2D", "experimental", "cp_test.txt"))
data_num_cp_2d = generate_data_num_cp(os.path.join(windtunnel_folder, "2D", "numerical", "cp"))
generate_plot_cp(data_exp_cp_2d, os.path.join(windtunnel_folder, "plots", "2D", "experimental", "cp"), "2D_exp_cp")
generate_plot_cp(data_num_cp_2d, os.path.join(windtunnel_folder, "plots", "2D", "numerical", "cp"), "2D_num_cp")
generate_plot_comp_cp(data_exp_cp_2d, data_num_cp_2d, os.path.join(windtunnel_folder, "plots", "2D", "comparison", "cp"), "2D_comp_cp")

data_exp_coeff_2d = generate_data_exp_coeff(os.path.join(windtunnel_folder, "2D", "experimental", "press_test.txt"))
data_num_coeff_2d = generate_data_num_coeff(os.path.join(windtunnel_folder, "2D", "numerical", "NACA642A015_T1_Re0.646_M0.12_N9.0.txt"))
generate_plot_coeff(data_exp_coeff_2d, os.path.join(windtunnel_folder, "plots", "2D", "experimental"), "2D_exp")
generate_plot_coeff(data_num_coeff_2d, os.path.join(windtunnel_folder, "plots", "2D", "numerical"), "2D_num")
generate_plot_comp_coeff(data_exp_coeff_2d, data_num_coeff_2d, os.path.join(windtunnel_folder, "plots", "2D", "comparison"), "2D_comp")



# Generate 3D plots
generate_plot_thermal(os.path.join(windtunnel_folder, "3D", "experimental", "thermal"), os.path.join(windtunnel_folder, "plots", "3D", "experimental", "thermal"), "3D_thermal")

# 3D numerical pressure distribution is not done yet
data_exp_cp_3d = generate_data_exp_cp(os.path.join(windtunnel_folder, "3D", "experimental", "cp_test.txt"))
data_num_cp_3d = generate_data_num_cp(os.path.join(windtunnel_folder, "3D", "numerical", "cp"))
generate_plot_cp(data_exp_cp_3d, os.path.join(windtunnel_folder, "plots", "3D", "experimental", "cp"), "3D_exp_cp")
generate_plot_cp(data_num_cp_3d, os.path.join(windtunnel_folder, "plots", "3D", "numerical", "cp"), "3D_num_cp")
generate_plot_comp_cp(data_exp_cp_3d, data_num_cp_3d, os.path.join(windtunnel_folder, "plots", "3D", "comparison", "cp"), "3D_comp_cp")

data_exp_coeff_3d = generate_data_exp_coeff(os.path.join(windtunnel_folder, "3D", "experimental", "press_test.txt"))
data_num_coeff_3d = generate_data_num_coeff(os.path.join(windtunnel_folder, "3D", "numerical", "NACA 642A015_T1_Re0.646_M0.12_N9.0.txt"))
generate_plot_coeff(data_exp_coeff_3d, os.path.join(windtunnel_folder, "plots", "3D", "experimental"), "3D_exp")
generate_plot_coeff(data_num_coeff_3d, os.path.join(windtunnel_folder, "plots", "3D", "numerical"), "3D_num")
generate_plot_comp_coeff(data_exp_coeff_3d, data_num_coeff_3d, os.path.join(windtunnel_folder, "plots", "3D", "comparison"), "3D_comp")



# Generate 2D - 3D comparison plots
generate_plot_comp_2d3d_coeff(data_exp_coeff_2d, data_num_coeff_2d, data_exp_coeff_3d, data_num_coeff_3d, os.path.join(windtunnel_folder, "plots", "comparison"), "2D3D_comp")




end = timeit.default_timer()
time = end - start
print("Total Processing time:", time, "seconds (" + str(round(time/60, 2)) + " mins) \n")