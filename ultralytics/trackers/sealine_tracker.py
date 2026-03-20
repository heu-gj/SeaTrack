# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils import matching_sealine


class SealineTracker(BYTETracker):
    """ByteTrack variant with adaptive IoU and scale-aware geometric association for sea-surface tracking."""

    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args=args, frame_rate=frame_rate)
        self.frame_shape: tuple[int, int] | None = None

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update tracker state using adaptive IoU, sea geometry cost, and modular recovery matching."""
        self.frame_id += 1
        if img is not None:
            self.frame_shape = img.shape[:2]

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = self.get_dists(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        remaining_tracked = [r_tracked_stracks[i] for i in u_track]
        recovery_detections = [detections[i] for i in u_detection]
        if self.args.use_recovery_matching:
            remaining_tracked, recovery_detections, activated_recovery, refind_recovery = self.recovery_match(
                remaining_tracked, recovery_detections
            )
            activated_stracks.extend(activated_recovery)
            refind_stracks.extend(refind_recovery)

        tracked_ids_after_recovery = {track.track_id for track in remaining_tracked}
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.track_id not in tracked_ids_after_recovery:
                continue
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = recovery_detections
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate association costs with innovation point 1 and innovation point 2."""
        dists = matching_sealine.sgl_distance(tracks, detections, self.frame_shape, self.args)
        if self.args.fuse_score:
            dists = matching_sealine.fuse_score(dists, detections)
        return dists

    def recovery_match(self, tracks: list[STrack], detections: list[STrack]):
        """Recovery stage for innovation point 3; kept modular for future replacement."""
        if not tracks or not detections:
            return tracks, detections, [], []

        dists = matching_sealine.sea_recovery_distance(tracks, detections, self.frame_shape, self.args)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.recovery_match_thresh)

        activated = []
        refind = []
        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        remaining_tracks = [tracks[i] for i in u_track]
        remaining_detections = [detections[i] for i in u_detection]
        return remaining_tracks, remaining_detections, activated, refind


__all__ = ("SealineTracker",)
