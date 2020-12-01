 set -e

#
# Data preprocessing configuration
#

N_MONO=3920000  # number of monolingual sentences for each language
CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono_qa_new
PARA_PATH=$DATA_PATH/para_qa_new

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$MONO_PATH/all.cl
TGT_RAW=$MONO_PATH/all.qu
SRC_TOK=$MONO_PATH/all.cl.tok  # tokenized cloze corpus file name
TGT_TOK=$MONO_PATH/all.qu.tok  # tokenized question corpus file name
BPE_CODES=$MONO_PATH/bpe_codes
CONCAT_BPE=$MONO_PATH/all.cl-qu.$CODES
SRC_VOCAB=$MONO_PATH/vocab.cl.$CODES
TGT_VOCAB=$MONO_PATH/vocab.qu.$CODES
FULL_VOCAB=$MONO_PATH/vocab.cl-qu.$CODES

# TODO: change this
#SRC_VALID=$PARA_PATH/dev.cl.tok
#TGT_VALID=$PARA_PATH/dev.qu.tok
SRC_TEST=$DATA_PATH/clozes/dev1.cl.tok
#TGT_TEST=$PARA_PATH/dev.qu.tok

#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


echo "Applying BPE to test files..."
$FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_TEST.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES