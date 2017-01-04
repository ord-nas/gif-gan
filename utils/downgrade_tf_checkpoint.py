import tensorflow as tf
import os
import sys
import argparse

parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--input_directory", required=True, help="Directory containing input checkpoint directory")
parser.add_argument("--output_file", required=True, help="Name of output checkpoint file")

def main():
    args = parser.parse_args()

    # Start a TF sessions
    sess = tf.Session()

    # Load the input model
    ckpt = tf.train.get_checkpoint_state(args.input_directory)
    if not ckpt or not ckpt.model_checkpoint_path:
        print "ERROR: couldn't load input checkpoint directory"
        sys.exit(1)

    # We do this because the model_checkpoint_path is absolute, and if it's
    # copied to somewhere new we actually want a relative path from the
    # checkpoint directory.
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    ckpt_path = os.path.join(args.input_directory, ckpt_name)

    print "Reading model from", ckpt_path

    # Restore the old model
    saver_v2 = tf.train.import_meta_graph(ckpt_path + ".meta")
    saver_v2.restore(sess, ckpt_path)

    # Make output directory if it doesn't exist
    output_directory = os.path.dirname(args.output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the new model
    initialized_vars = [v for v in tf.global_variables()
                        if sess.run(tf.is_variable_initialized(v))]
    saver_v1 = tf.train.Saver(write_version=1, var_list=initialized_vars)
    saver_v1.save(sess, args.output_file, global_step=0)

    print "Successfully wrote model to", args.output_file

if __name__ == "__main__":
    main()
