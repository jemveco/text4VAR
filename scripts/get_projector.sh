if [ -z $1 ]; then
    cat << EOF
ERROR: Need a \`list.yaml\` file
 -> Usage: get_projector.sh [--yaml <YAML file> -s | --single ]
 -> Note:
     1) The \`--yaml\` flag is optional, if not set the
        first argument will be taken as the yaml file.
     2) The \`-s\` (or \`--single\`) flag is also optional,
        if not set the script will launch \`projector.py\`,
        \`projector_single_description.py\` will be launched
        otherwise.
EOF
    exit 1
fi

FIRST_LOOP=
YAML=
SINGLE_PROJECTOR=
while [ $# -gt 0 ]; do
    case "$1" in
        "--yaml" )
            YAML=$2
            shift 2
            ;;
        "-s"|"--single" )
            SINGLE_PROJECTOR=1
            shift
            ;;
        * )
            if [ -r $1 ] && [ -z "$FIRST_LOOP" ]; then
                YAML=$1
                FIRST_LOOP="no"
            fi
            shift
            ;;
    esac
done
 
MODEL="ViT-L/14"        # Model to extract the text embeddings (for ViT models it uses the text encoder of CLIP)
REDUCTION="no_reduce"         # Reduction method
DIMENSION=192           # The dimension to which reduce the embedding with the `REDUCTION`
CENTROID="mean"         # The method to compute the centroids
TRUNCATE=1              # If the input to the model should be truncated 

cat << EOF
Generating projector with:
 -> Yaml: $YAML
 -> single: $SINGLE_PROJECTOR
 -> Model: $MODEL
 -> Reduction method: $REDUCTION
 -> Dimension: $DIMENSION
 -> Centroid: $CENTROID
 -> Truncate: $TRUNCATE
EOF

if [ -z $SINGLE_PROJECTOR ]; then
    python projector.py --yaml $YAML --model $MODEL --reduction-method $REDUCTION \
--dimension $DIMENSION --centroid-method $CENTROID --truncate-input $TRUNCATE
else
    python projector_single_description.py --yaml $YAML --model $MODEL \
--reduction-method $REDUCTION --dimension $DIMENSION --truncate-input $TRUNCATE
fi

