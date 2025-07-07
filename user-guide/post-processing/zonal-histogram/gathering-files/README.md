---
description: >-
  We need to find the folders that the labels we generated live in so that SIT
  FUSE knows where to look. So we first must find all of the unique computer
  paths to all of our shape-files (labels).
icon: people-arrows
---

# Gathering files

When we created the labels using QGIS, we chose where on the computer/what folder(s) to store them in. We are using the terminal to find them, rather than looking, for example, on our Desktop. That is because we need the exact computer path for each file, and terminal makes that easy to find.

**First, make sure you have your empty YAML file. Then, copy the first cell block from the** [**Code configuration**](code-configuration.md) **page, removing the descriptions after the colons once we start inputting files. With that, we are ready to start finding our files.**

Here's an easy way to locate any path(s)/directory:

1. Open Terminal.
2.  Navigate to the directory(s) where your **shape-files** are stored.

    E.g. **If** you had a folder called shape\_files on your Desktop, you would type `cd Users/<your_username>/Desktop/shape_files/` (replacing `<your_username>` with your username).&#x20;

_Hint: type `ls` to list all files and directories in your current location._

_Hint 2: type `pwd` to see your current location._

3. Once you are at the desired directory, locate all .shp files and their respective paths by typing `find $PWD -name "*.shp" | sort` in the terminal.&#x20;

* `$PWD` shows us the full path before the name of each file (otherwise we'd only see the file name).&#x20;
* `-name "*.shp"` finds every filename ending `.shp` (AKA every label).
* The `*` is a wildcard, meaning we are telling the computer that we don't care what comes before `.shp`, as long as the file has `.shp` at the end of the name.
* `sort` sorts the files alphabetically, which is critical for the organization of our YAML.

<figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-24 at 9.14.29 PM.png" alt=""><figcaption><p>Example output for step 3</p></figcaption></figure>

Note: Ideally your labels are organized with respect to the exact image/data/gtiff they labeled (aka they're organized/separated by folders, as seen above with `MASTER1, MASTER10, MASTER11, MASTER2,...etc`). We want the labels for each specific image to be grouped together in order for the YAML to read them properly. See the `Labeler's Note` in [Creating and modifying shapefiles](../../labeling-tutorial/creating-and-modifying-shapefiles.md) after step 7.

4. Copy the desired output and input it into the `shp_files` configuration of your YAML file. Don't worry about formatting right now, we'll see some examples in the [next page](code-configuration.md). Just make sure every label you want is included, and try to ensure they are grouped in the way described in the above note.&#x20;
5. Repeat this process for `input_files` , which would look something like `find $PWD -name "*.tif" | sort` . Make sure every GeoTiff that you want is included.

_Important note: For an overview of linux and the command line, click_ [_here_](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)_. For more documentation on finding files in linux, click_ [_here_](https://www.geeksforgeeks.org/find-command-in-linux-with-examples/)_. It's important to be comfortable working in the terminal as we will use it a lot throughout the context assignment process._&#x20;
