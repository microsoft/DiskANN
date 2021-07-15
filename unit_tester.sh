#!/bin/sh

if [ "$#" -ne "2" ]; then
  echo "usage: ./unit_test.sh [path_build_folder] [working_folder]"
else

BUILD_FOLDER=${1}
WORK_FOLDER=${2}

echo Running unit testing on various files, with build folder as ${BUILD_FOLDER} and working folder as ${WORK_FOLDER}
# download all unit test files

#iterate over them and run the corresponding test
CATALOG1="${WORK_FOLDER}/catalog.txt"
CATALOG="${WORK_FOLDER}/catalog_formatted.txt"
sed -e '/^$/d' ${CATALOG1} > ${CATALOG}

mkdir ${WORK_FOLDER}/indices

while IFS= read -r line; do
  DATASET=${line}
  BASE="${WORK_FOLDER}/${DATASET}"
  read -r line
  QUERY="${WORK_FOLDER}/${line}"
  read -r TYPE
  read -r METRIC
  GT="${WORK_FOLDER}/indices/${DATASET}_gt30_${METRIC}"
  MEM="${WORK_FOLDER}/indices/${DATASET}_mem"
  DISK="${WORK_FOLDER}/indices/${DATASET}_disk"
  MBLOG="${WORK_FOLDER}/indices/${DATASET}_mb.log"
  DBLOG="${WORK_FOLDER}/indices/${DATASET}_db.log"
  MSLOG="${WORK_FOLDER}/indices/${DATASET}_ms.log"
  DSLOG="${WORK_FOLDER}/indices/${DATASET}_ds.log"
  echo "Going to run test on ${BASE} base, ${QUERY} query, ${TYPE} datatype, ${METRIC} metric, saving gt at ${GT}"
  echo "Computing Groundtruth"
  ${BUILD_FOLDER}/tests/utils/compute_groundtruth ${TYPE} ${BASE} ${QUERY} 30 ${GT} ${METRIC} > /dev/null
  echo "Building Mem Index"
  ${BUILD_FOLDER}/tests/build_memory_index ${TYPE} ${METRIC} ${BASE} ${MEM}  32  50  1.2 0 > ${MBLOG}
  awk '/^Degree/' ${MBLOG}
  awk '/^Indexing/' ${MBLOG}
  echo "Building Disk Index"
  ${BUILD_FOLDER}/tests/build_disk_index  ${TYPE} ${METRIC} ${BASE} ${DISK} 32 50 0.03 0.01 32 0 > ${DBLOG}
  awk '/^Compressing/' ${DBLOG}
  echo "#shards in disk index"
  awk '/^bin:/' ${DBLOG}
  awk '/^Indexing/' ${DBLOG}
  echo "Searching Mem Index"
  ${BUILD_FOLDER}/tests/search_memory_index ${TYPE} ${METRIC} ${BASE} ${MEM} 16 ${QUERY} ${GT} 10 /tmp/res 10 20 30 40 50 60 70 80 90 100 > ${MSLOG}
  awk '/===/{x=NR+10}(NR<=x){print}' ${MSLOG}
  echo "Searching Disk Index"
  ${BUILD_FOLDER}/tests/search_disk_index ${TYPE} ${METRIC} ${DISK} 10000 10 4 ${QUERY} ${GT} 10 /tmp/res 10 20 30 40 50 60 70 80 90 100 > ${DSLOG}
  awk '/===/{x=NR+10}(NR<=x){print}' ${DSLOG}
done < "${CATALOG}"

fi
