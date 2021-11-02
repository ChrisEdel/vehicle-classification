## Install instructions of the preprocessing/data augmentation program called ``vis_c`` on Ubuntu

1. Run ``pcl_vis_c_setup.sh`` to install PCL and ``vis_c`` dependenc.
2. Create directory ``build`` in the same directory as the files ``main.cpp`` and ``CMakeLists.txt``.
3. Go into the ``build`` directory with ``cd build`` and execute ``cmake ..``, then ``make``.
4. Run ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib``.
5. Execute ``./vis_c`` to check, if it is working.

    Optional:

6. To make ``vis_c`` available on the entire machine execute ``sudo cp vis_c /usr/local/bin``.
7. To make step 4 permanent add ``LD_LIBRARY_PATH=/usr/local/lib`` to the file ``/etc/environment`` or extend the entry, if ``LD_LIBRARY_PATH`` is already in there; you need to log out and back in for the change to take effect.
