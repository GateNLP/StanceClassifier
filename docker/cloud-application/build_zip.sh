#!/bin/sh

if [ -z "$3" ]; then
  echo "Usage: ./build_zip.sh <metadata_dir> <endpoint_url> <output_file>"
  echo ""
  echo "<metadata_dir>: directory for the metadata, e.g. metadata/english"
  echo "<endpoint_url>: URL of the ELG endpoint to which we will POST."
  echo "   For services in the GATE Cloud cluster this is along the lines of"
  echo "   http://stance-english.elg.svc.cluster.local/process"
  echo "<output_file>: name for the output zip file to create in this dir"
  exit
fi

MD_DIR=$1
ENDPOINT=$2
OUTFILE=$3

mkdir build
sed s#@ELG_ENDPOINT@#$ENDPOINT# application.xgapp > build/application.xgapp
cp -a plugins build/
cp -a $MD_DIR build/metadata

cd build
zip -r ../$OUTFILE *
cd ..
rm -rf build
