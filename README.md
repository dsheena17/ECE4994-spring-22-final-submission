# ECE4994-spring-22-final-submission
My final submission for ECE 4994

This repository consists of 3 folders with 3 versions of the same code: 
 1. float: has the original code with IEEE standard float types. This is meant to be a reference in comparision with the other two files. 
 2. ac_fixed: uses ac_fixed types instead of floats.
 3. hls_float: uses hls_float types instead of regular floats.

Both, options 2 and 3 conserve FPGA area as they reduce the number of bits required for each arithmetic computation, and over multiple iterations, it adds up. 

To compile the files, navigate to their respective directory and run the following command: 
> i++ -v filename.cpp -march=CycloneV -o projectfilename

This should generate executable files and a project folder with all the reqired files. 

Notes: 
1. I had issues running the files in directories without administrative permissions, but changing permissions worked for me. 
2. The code curretly rails towards infinity or 0 NRMSE occassionally and I am not sure why at the moment. This is something I intend to continue working on. 
3. The files might be too large to run on CycloneV's processor, and may not generate all the files (including \*.wlf files required for questa simulations). 
4. The number of samples was reduced from 4000 and 2000 initial and training samples to 400 and 200 respectively to be able to run the executable. 
5. Similarly, the number of nodes was reduced from 400 to 40. This results in a different NRMSE.

This project is still a work in progress, so updates will be made in the future!
