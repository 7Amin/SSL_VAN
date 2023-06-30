# SSL_VAN


LUNA16: LUng Nodule Analysis 2016
https://luna16.grand-challenge.org/

We excluded scans with a slice thickness greater than 2.5 mm. In total, 888 CT scans are included (843 training - 45 validation)
The data contains annotations which were collected during a two-phase annotation process using 4 experienced radiologists
Each radiologist marked lesions they identified as non-nodule, nodule < 3 mm, and nodules >= 3 mm.
The reference standard of our challenge consists of all nodules >= 3 mm accepted by at least 3 out of 4 radiologists.

(0020, 1041) Slice Location                      DS: '-69.5' is <imageZposition>-69.5</imageZposition> in the xml file

conda create -n ssl_van_seg python=3.9

SSL_VAN]$ sbatch nodule_segmentor/jobs/job.sh

BTCV
http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data.zip
http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg.zip
http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg-canine-legs.tar.bz2



You may download the labeled training and unlabeled testing data from: 

http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data.zip (4.3 GB)

The registered labeled (training) data for deep brain structures are located here: 
http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg.zip  (23.9 GB)

The registered labeled (training) data for canine leg structures are located here: 
http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg-canine-legs.tar.bz2   (73 GB)

New notes - 6/19/2013. 
1.	Here is the updated CAP data. The testing data has been corrected http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg-CAP-v2.tar.bz2 

2.	There are 15 files total: http://masi.vuse.vanderbilt.edu/MICCAI-2013-SATA-Challenge-Data-Std-Reg-canine-legs.tar.bz2-part-a[a-o]
a.	The proper way to reconstruct the full tar.bz2 is through:
"cat  MICCAI-2013-SATA-Challenge-Data-Std-Reg-canine-legs.tar.bz2-part-a* > MICCAI-2013-SATA-Challenge-Data-Std-Reg-canine-legs.tar.bz2"


export MASTER_ADDR=localhost
export MASTER_PORT=12345
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export LD_LIBRARY_PATH=/home/karimimonsefi.1/miniconda3/envs/ssl_van_seg/lib:$LD_LIBRARY_PATH

