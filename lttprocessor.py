import os, timeit, operator
import numpy
from matplotlib import pyplot, transforms, ticker
from PIL import Image, ImageDraw

def generate_plot_thermal(dirpath):
    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]

    # Declare temperature array
    temperature_array = 0

    # Keep count of how many files have been treated, also needed to create array and add subsequent readings to first array
    filecounter = 1

    for file in files:
        # If file is csv file, then use it
        test = file.rsplit(".", 1)
        if (test[-1] == "csv"):
            path = os.path.join(dirpath, file)
            print(f"Processing {path}")

            # If file is first file that is read
            if not (filecounter > 1):
                # Read the csv file and create array from it
                temperature_array = numpy.genfromtxt(os.path.join(root, file), dtype = numpy.float, delimiter = ";")
            else:
                break
                # Read the csv file and add array results to previous ones
                add = numpy.genfromtxt(os.path.join(root, file), dtype = numpy.float, delimiter = ";")
                temperature_array = temperature_array + add

            filecounter += 1

    if filecounter > 1:
        # Average temperatures
        #temperature_array = temperature_array / filecounter

        # Create mask to cut out wing
        wingshape = [(209,0),(485,0),(470,472),(404,476),(470,472),(324,472),(199,469)]
        #wingshape = [(300,0),(320,0),(320,20),(300,20)]
        x_min = min(wingshape)[0]
        x_max = max(wingshape)[0]
        y_min = min(wingshape,key=operator.itemgetter(1))[1]
        y_max = max(wingshape,key=operator.itemgetter(1))[1]

        img_wingmask = Image.new('L', (temperature_array.shape[1], temperature_array.shape[0]), 0)
        ImageDraw.Draw(img_wingmask).polygon(wingshape, outline=1, fill=1)
        wingmask = numpy.array(img_wingmask)


        masked_temperature_array = numpy.ma.array(temperature_array, mask = numpy.logical_not(wingmask))
        trimmed_masked_temperature_array = numpy.fliplr(masked_temperature_array[y_min:y_max, x_min:x_max])

        #tr = transforms.Affine2D().rotate_deg(0)

        pyplot.imshow(trimmed_masked_temperature_array, cmap = 'jet')
        cb = pyplot.colorbar()
        cb.set_label('°C', rotation = 0)
        #pyplot.show()
        ax = pyplot.gca()
        #ax.set_xticks(numpy.arange(x_min, x_max, 25))
        #ax.set_xticklabels(numpy.arange(0, 100, 25))
        
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(x_max - x_min))
        ax.xaxis.set_major_locator(ticker.LinearLocator(5))
        
        ax.set_xlabel("Chord")
        

        pyplot.savefig(os.path.join(root, f"thermal_{os.path.basename(os.path.normpath(root))}.png"), transparent = True, dpi = 300)
        pyplot.clf()

            


def generate_plot_cp(dirpath):
    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    
    for file in files:
        # If file is csv file, then use it
        if (file == "cp_test.txt"):
            path = os.path.join(root, file)
            print(f"Processing {path}")
            data = numpy.genfromtxt(path, "\t")

            # Plotting example of a pressure distribution (cp file)
            # Reynolds number can be accessed with data[dataset, 0]
            # Angle of Attack can be accessed with data[dataset, 1]

            # Produce graph of all datasets in the file
            targetpath = os.path.join(root, "plots", "Cp_AoA")
            for dataset in range(1, data.shape[0]):
                    
                if not os.path.exists(targetpath):
                    os.makedirs(targetpath)

                pyplot.xlabel(data[0][1])
                pyplot.ylabel("C_p")
                pyplot.plot(data[0, 2:].astype(numpy.float, casting = "unsafe"), data[dataset, 2:].astype(numpy.float, casting = "unsafe"))
                Re = data[dataset, 0].astype(numpy.int, casting = "unsafe")
                AoA = int(round(data[dataset, 1].astype(numpy.float, casting = "unsafe")))
                pyplot.savefig(os.path.join(targetpath, f"Re{Re}_AoA{AoA}.png"))
                pyplot.clf()
    
def generate_plot_sensor(dirpath):
    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    
    for file in files:
        # If file is csv file, then use it
        if (file == "press_test.txt"):
            path = os.path.join(root, file)
            print(f"Processing {path}")
            data = numpy.genfromtxt(os.path.join(root, file), "\t")

            # Plotting example of data in the press file
            y_dataset = 4 # Column number of the desired data set in the press file, for example 4 when y-axis is lift coefficient
            x_dataset = 2 # Column number of the desired data set in the press file, for example 2 when x-axis is Angle of Attack
                
            targetpath = os.path.join(root, "plots")
            if not os.path.exists(targetpath):
                os.makedirs(targetpath)

            pyplot.xlabel(data[x_dataset - 1][0])
            pyplot.ylabel(data[y_dataset - 1][0])
            pyplot.plot(data[x_dataset - 1, 2:].astype(numpy.float, casting = "unsafe"), data[y_dataset - 1, 2:].astype(numpy.float, casting = "unsafe"))#
            pyplot.savefig(os.path.join(targetpath, f"{data[y_dataset - 1][0]}_{data[x_dataset - 1][0]}.png"))
            pyplot.clf()
      
      







windtunnel_folder = "D:\courses\windtunnel\AE2130-II G25"

start = timeit.default_timer()

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(windtunnel_folder):
    #path = root.split(os.sep)
    #print((len(path) - 1) * '/', os.path.basename(root))
    #print(root)

    generate_plot_thermal(root)
    #generate_plot_cp(root)
    #generate_plot_sensor(root)

    #for file in files:
    #    print(len(path) * '/', file)

end = timeit.default_timer()
time = end - start
print("Total Processing time:", time, "seconds (" + str(round(time/60, 2)) + " mins) \n\n")
        