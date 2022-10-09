cd build && sudo make -j \
&& cd ../python \
&& rm -rf vamanapy* && rm -rf build && \
pip install -e .
