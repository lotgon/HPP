# HPP
Heterogeneous parallel programming

Test the same simple code on CPU, Cuda with/without tiling, C++ AMP with/without tiling, Cudafy

Results
 - Tiling is very difficult to implement. Could not implement it better than sequential algorithm. It seems gpu cache works better
 - C++ Amp works 5.7 seconds for test, CUDA 6.1 sec, Cudafy 7.5 sec. Hardware: Geforce 750i, 5.0 comp, i7
 - There are a lot of crashes from c++ amp and cudafy, cuda crashes happens only several times :) with BIG data.. You should allocate your memory very careful. It seems array_view doesn`t do it correctly.
 - C++ amp was the best option for me. Unfortunately, their forum is almost dead and no new updates. I was impressed that c++ compiler show races in my code. 
 - CUDA - it was very strange that creating temporary variable(caching the result of calculation) improve performance of application. Compiler should optimize it itself.
 
 
