import sys

import track_faces
from track_faces import Stats

class EmptyArgs:
    pass

def combine_histograms(a, b):
    combo = dict(a)
    for (k, v) in b.iteritems():
        combo[k] = v + combo.get(k, 0)
    return combo

def combine_two_stats(a, b):
    combo = Stats()

    # Drop counters (in priority order, higher to lower)
    combo.cnt_drop_because_low_frame_count = a.cnt_drop_because_low_frame_count + b.cnt_drop_because_low_frame_count
    combo.cnt_drop_because_low_total_detections = a.cnt_drop_because_low_total_detections + b.cnt_drop_because_low_total_detections
    combo.cnt_drop_because_expanded_bb_too_big = a.cnt_drop_because_expanded_bb_too_big + b.cnt_drop_because_expanded_bb_too_big
    combo.cnt_drop_because_optical_flow_bb_too_big = a.cnt_drop_because_optical_flow_bb_too_big + b.cnt_drop_because_optical_flow_bb_too_big
    combo.cnt_drop_because_no_feature_points = a.cnt_drop_because_no_feature_points + b.cnt_drop_because_no_feature_points
    combo.cnt_drop_because_failed_optical_flow = a.cnt_drop_because_failed_optical_flow + b.cnt_drop_because_failed_optical_flow
    combo.cnt_drop_because_no_rigid_transform = a.cnt_drop_because_no_rigid_transform + b.cnt_drop_because_no_rigid_transform
    combo.cnt_drop_because_stabilized_bb_too_big = a.cnt_drop_because_stabilized_bb_too_big + b.cnt_drop_because_stabilized_bb_too_big

    # Truncate counters (in priority order, higher to lower)
    combo.cnt_truncate_because_optical_flow_bb_too_big = a.cnt_truncate_because_optical_flow_bb_too_big + b.cnt_truncate_because_optical_flow_bb_too_big
    combo.cnt_truncate_because_no_feature_points = a.cnt_truncate_because_no_feature_points + b.cnt_truncate_because_no_feature_points
    combo.cnt_truncate_because_failed_optical_flow = a.cnt_truncate_because_failed_optical_flow + b.cnt_truncate_because_failed_optical_flow
    combo.cnt_truncate_because_no_rigid_transform = a.cnt_truncate_because_no_rigid_transform + b.cnt_truncate_because_no_rigid_transform
    combo.cnt_truncate_because_stabilized_bb_too_big = a.cnt_truncate_because_stabilized_bb_too_big + b.cnt_truncate_because_stabilized_bb_too_big

    # Other counters
    combo.cnt_total_detections = a.cnt_total_detections + b.cnt_total_detections
    combo.cnt_detections_kept = a.cnt_detections_kept + b.cnt_detections_kept
    combo.cnt_detections_written = a.cnt_detections_written + b.cnt_detections_written
    combo.cnt_initial_tracks = a.cnt_initial_tracks + b.cnt_initial_tracks
    combo.cnt_final_tracks = a.cnt_final_tracks + b.cnt_final_tracks
    combo.cnt_errors = a.cnt_errors + b.cnt_errors
    combo.cnt_input_files = a.cnt_input_files + b.cnt_input_files
    combo.cnt_processed_files = a.cnt_processed_files + b.cnt_processed_files
    combo.cnt_tracks_with_interpolation = a.cnt_tracks_with_interpolation + b.cnt_tracks_with_interpolation

    # Histograms
    combo.hst_skip_raw = combine_histograms(a.hst_skip_raw, b.hst_skip_raw)
    combo.hst_skip_used = combine_histograms(a.hst_skip_used, b.hst_skip_used)
    combo.hst_jaccard_raw = combine_histograms(a.hst_jaccard_raw, b.hst_jaccard_raw)
    combo.hst_jaccard_used = combine_histograms(a.hst_jaccard_used, b.hst_jaccard_used)
    combo.hst_track_len_raw = combine_histograms(a.hst_track_len_raw, b.hst_track_len_raw)
    combo.hst_track_len_used = combine_histograms(a.hst_track_len_used, b.hst_track_len_used)
    combo.hst_num_feature_points = combine_histograms(a.hst_num_feature_points, b.hst_num_feature_points)
    combo.hst_num_tracked_feature_points = combine_histograms(a.hst_num_tracked_feature_points, b.hst_num_tracked_feature_points)
    combo.hst_time = combine_histograms(a.hst_time, b.hst_time)
    combo.hst_frame_height_raw = combine_histograms(a.hst_frame_height_raw, b.hst_frame_height_raw)
    combo.hst_frame_width_raw = combine_histograms(a.hst_frame_width_raw, b.hst_frame_width_raw)
    combo.hst_frame_height_used = combine_histograms(a.hst_frame_height_used, b.hst_frame_height_used)
    combo.hst_frame_width_used = combine_histograms(a.hst_frame_width_used, b.hst_frame_width_used)
    combo.hst_num_crops_per_video = combine_histograms(a.hst_num_crops_per_video, b.hst_num_crops_per_video)

    combo.start_time = min(a.start_time, b.start_time)
    combo.start_time_str = a.start_time_str if combo.start_time == a.start_time else b.start_time_str

    return combo

def combine_stats(stats):
    return reduce(combine_two_stats, stats)

def combine_stats_files(filenames, output_txt, output_html):
    stats = []
    for n in filenames:
        with open(n, 'r') as f:
            d = eval(f.read())
        s = Stats()
        s.__dict__.update(d)
        stats.append(s)
    combo = combine_stats(stats)
    args = EmptyArgs()
    track_faces.write_stats_to_files(args, combo, output_txt, output_html)

def main():
    args = sys.argv
    if len(args) < 4:
        print "Usage: python stat_combination.py in_1 [in_2 [in_3 [in_4 ...]]] out_txt out_html"
        exit(1)

    output_html = args[-1]
    output_txt = args[-2]
    filenames = args[1:-2]

    combine_stats_files(filenames, output_txt, output_html)


if __name__ == "__main__":
    main()
