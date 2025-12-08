#!/bin/bash

function HELP {
    cat <<HELP


--------------------------------   yhBrainExt.sh  ------------------------------------
  The script for the best brain extraction using ANTs (http://stnava.github.io/ANTs/).

  Description:
  This script is modified using previous antsBrainExtraction.sh script and other
  ants commands.
  The script is very well working in human brains regardless of shapes and ages.
  However, it is not perfectly working sometimes in B1-bias field affected brains.
  For example, too much extraction in temporal cortex in 7 Tesla T1w images.

  Development History:
    Version 0.41: added soft brain extraction option using new approach (2020.4.26)
    Version 0.4 : changed strategy (antsBrainExtraction) (2020.2.3)
    Version 0.33: resolution matched / initial ANTs registration
                  (very sensitivy to interpolation especially in mprage) (2020.1.15)
    Version 0.32: soft erosion of the result image (2019.7.29)
    Version 0.31: added bet algorithm to clean up final result (2019.3.26)
    Version 0.3 : Improved performance (2019.3.15)
    Version 0.21: refine input image (2018.9.6)
    Version 0.20: changed strategy (2018.8.24)
    Version 0.10: the script release (2018.7.20)

--------------------------------------------------------------------------------------
  Example Usage:
  yhBrainExt.sh -i input -o input_Brain -s soft -p 1

  (Optional)
  yhBrainExt.sh -version (see the version)

  Compulsory arguments:
      -i:  input image (nifti file)
      -o:  final output name
      -s:  select extraction amount (hard / soft)
      -p:  intermediate files (1: remain / 0: remove)

--------------------------------------------------------------------------------------
  Requirement: ANTs pre-installation / FSL pre-installation
--------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------
  This method was created by:

  Uksu, Choi (uschoi@nict.go.jp)
  Center for Information and Neural Networks
  National Institute of Information and Communications Technology

--------------------------------------------------------------------------------------
                      Script writing and modification by Uksu
                      Do not modify without a permission.
--------------------------------------------------------------------------------------


HELP
    exit 1
}

# reading command line arguments
while getopts "h:i:o:s:p:v:" OPT
do
  case $OPT in
      h)
      HELP
   exit 0
   ;;
      i)
      moving=$OPTARG
      ;;
      o)
      moving_output=$OPTARG
      ;;
      s)
      ext=$OPTARG
      ;;
      p)
      residual=$OPTARG
      ;;
      v)
      version=1
      ;;
      \?) # getopts issues an error message
   echo $USAGE >&2
   exit 1
   ;;
 esac
done

if [[ ${moving: -7} == ".nii.gz" ]];
then
echo "
Please remove and an input file extension (nii.gz) \

"
exit 1
fi

if [[ ${moving_output: -7} == ".nii.gz" ]];
then
echo "
Please remove an output file extension (nii.gz) \

"
exit 1
fi

if [[ ! -z ${version} ]];then
  echo "
  Current version is 0.41 \

  "
  exit 1
fi

########################################### preprocessing ####################################################################
# time_start
time_start=`date +%s`

# refine input image
fslreorient2std ${moving} ${moving}

# bias field correction
# echo "Now bias correction is starting ..."
N4BiasFieldCorrection -d 3 -i ${moving}.nii.gz -b [200] -o ${moving}_bias.nii.gz -v
# echo "Now bias correction finished"
#
# # optional bet
# bet ${moving}_bias ${moving}_bias -f 0.04
# fslmaths ${moving} ${moving}_bias
########################################### Extraction  ####################################################################
# Brain extraction with antsBrainExtraciton.sh
if [ "$ext" == "soft" ];then
  echo " ++ SOFT Extraction is starting ... "
  antsRegistration \
      --collapse-output-transforms 1 \
      --dimensionality 3 \
      --interpolation Linear \
      --output [Tmp, Tmp1Warp.nii.gz,Tmp1WarpInv.nii.gz] \
      --use-histogram-matching 1 \
      --winsorize-image-intensities [0.005,0.995] \
      --initial-moving-transform [${FSL_DIR}/data/standard/MNI152_T1_1mm.nii.gz,${moving}_bias.nii.gz, 0] \
      --transform Rigid[0.1] \
      --metric MI[${FSL_DIR}/data/standard/MNI152_T1_1mm.nii.gz,${moving}_bias.nii.gz,1,32,Regular,0.25] \
      --convergence [1000x500,1e-6,10] \
      --shrink-factors 12x8 \
      --smoothing-sigmas 4x3vox
  # change name
  cp Tmp1Warp.nii.gz ${moving}MNI_init.nii.gz

  ########################################### New Strategy  ####################################################################
  # Brain extraction with antsBrainExtraciton.sh
  antsBrainExtraction.sh \
                      -d 3 \
                      -a ${moving}MNI_init.nii.gz \
                      -e ${YH_TEMPLATE:-/data/data2/dataset/fpreproc/template/adult/T_template0.nii.gz} \
                      -m ${YH_PROB_MASK:-/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumProbabilityMask.nii.gz} \
                      -f ${YH_REG_MASK:-/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumRegistrationMask.nii.gz} \
                      -o Ext

  # Inverse transformation
  antsApplyTransforms \
            --dimensionality 3 \
            --input ExtBrainExtractionBrain.nii.gz \
            --reference-image ${moving}_bias.nii.gz \
            --output ${moving_output}.nii.gz \
            --n Linear \
            --transform [Tmp0GenericAffine.mat,1]

  # clean
  # rm *_out_tmp*
  rm -r Ext
elif [ "$ext" == "hard" ];then
  echo "++ HARD Extraction is starting ..."
  antsBrainExtraction.sh \
                      -d 3 \
                      -a ${moving}_bias.nii.gz \
                      -e ${YH_TEMPLATE:-/data/data2/dataset/fpreproc/template/adult/T_template0.nii.gz} \
                      -m ${YH_PROB_MASK:-/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumProbabilityMask.nii.gz} \
                      -f ${YH_REG_MASK:-/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumRegistrationMask.nii.gz} \
                      -o Ext
  # resolution info
  resol=($(echo `fslval ${moving}.nii.gz pixdim1`))
  fslmaths ExtBrainExtractionBrain.nii.gz -kernel gauss $resol -ero ${moving_output}.nii.gz
  # clean
  rm -r Ext
else
  echo " ++ !! Wrong Extraction option !! "
  exit 1
fi


########################################### Old Strategy  ####################################################################
# ## resolution match (updated)
# # flirt -in ${moving}_bias.nii.gz -ref ${moving}_bias.nii.gz -applyisoxfm 1 -out ${moving}_bias.nii.gz
# flirt -in ${moving}_bias.nii.gz -ref ${moving}_bias.nii.gz -interp trilinear -applyisoxfm 1 -out ${moving}_bias.nii.gz
#
# # linear antsRegistration
# # Deformation (NearestNeighbor/Linear)
# antsRegistration \
#     --collapse-output-transforms 1 \
#     --dimensionality 3 \
#     --interpolation Linear \
#     --output [Ext, Ext1Warp.nii.gz,Ext1WarpInv.nii.gz] \
#     --use-histogram-matching 1 \
#     --winsorize-image-intensities [0.005,0.995] \
#     --initial-moving-transform [${FSL_DIR}/data/standard/MNI152_T1_1mm.nii.gz,${moving}_bias.nii.gz, 0] \
#     --transform Rigid[0.1] \
#     --metric MI[${FSL_DIR}/data/standard/MNI152_T1_1mm.nii.gz,${moving}_bias.nii.gz,1,32,Regular,0.25] \
#     --convergence [1000x500,1e-6,10] \
#     --shrink-factors 12x8 \
#     --smoothing-sigmas 4x3vox
#
# # Brain extraction with antsBrainExtraciton.sh
# antsBrainExtraction.sh \
#                     -d 3 \
#                     -a Ext1Warp.nii.gz \
#                     -e /Users/uschoi/bin/adult/T_template0.nii.gz \
#                     -m /Users/uschoi/bin/adult/T_template0_BrainCerebellumProbabilityMask.nii.gz \
#                     -c 3x1x2x3 \
#                     -f /Users/uschoi/bin/adult/T_template0_BrainCerebellumRegistrationMask.nii.gz \
#                     -q 1 \
#                     -o Ext
#
# # Inversed transformation fixed to moving
# antsApplyTransforms \
#           --dimensionality 3 \
#           --input ExtBrainExtractionBrain.nii.gz \
#           --reference-image ${moving}.nii.gz \
#           --output ${moving_output}_tmp.nii.gz \
#           --n Linear \
#           --transform [Ext0GenericAffine.mat,1]
#
# # optional erosion or bet again
# # bet ${moving_output}.nii.gz ${moving_output}.nii.gz -f 0.03
# resol=($(echo `fslval ${moving_output}_tmp.nii.gz pixdim1`))
# fslmaths ${moving_output}_tmp.nii.gz -eroF -bin ero_mask
# fslmaths ${moving_output}_tmp.nii.gz -mas ero_mask ${moving_output}_tmp2.nii.gz
# fslmaths ${moving} -mas ${moving_output}_tmp2 ${moving_output}


####################################################################################################################################
if [ "$residual" == "1" ];then
echo "++ Keep intermediate files"
else
echo " ++ Removing intermediate files"
rm ${moving}_bias.nii.gz ExtBrainExtractionBrain.nii.gz ExtBrainExtractionMask.nii.gz \
ExtBrainExtractionPrior0GenericAffine.mat
fi

# time
time_end=`date +%s`
time_elapsed=$((time_end - time_start))
echo
echo "--------------------------------------------------------------------------------------"
echo " yhBrainExt process was completed in $time_elapsed seconds"
echo " $(( time_elapsed / 3600 ))h $(( time_elapsed %3600 / 60 ))m $(( time_elapsed % 60 ))s"
echo "--------------------------------------------------------------------------------------"
exit 0

####################################################################################################################################
