"""Aggregate actual PhysX collider contacts for inspectable scene interaction."""

import numpy as np


class ContactLedger:
    def __init__(self, physics_dt=0.005, *, floor_actor_paths=()):
        self.dt, self.pairs, self.errors = physics_dt, {}, []
        self.floor_actor_paths = frozenset(str(path) for path in floor_actor_paths)
        self.contact_states = {}

    def record(self, headers, data, decode_path, *, time, phase):
        for header in headers:
            count = header.num_contact_data
            actors = [str(decode_path(header.actor0)), str(decode_path(header.actor1))]
            colliders = [str(decode_path(header.collider0)), str(decode_path(header.collider1))]
            key = " | ".join(a + "::" + c for a, c in sorted(zip(actors, colliders)))
            event = getattr(header, "type", None)
            if event is not None:
                name = str(getattr(event, "name", event)).upper()
                if "CONTACT_LOST" in name or str(event) == "1":
                    active = False
                elif "CONTACT_FOUND" in name or "CONTACT_PERSIST" in name or str(event) in ("0", "2"):
                    active = True
                else:
                    active = None
                if active is not None:
                    self.contact_states[key] = dict(
                        actors=actors,
                        colliders=colliders,
                        active=active,
                        last_event=name,
                        last_event_time=time,
                        last_event_phase=phase,
                    )
            if not count:
                continue
            points = [data[i] for i in range(header.contact_data_offset, header.contact_data_offset + count)]
            impulse = sum(float(np.linalg.norm(tuple(c.impulse))) for c in points)
            if not np.isfinite(impulse):
                raise ValueError("Nonfinite contact impulse")
            if impulse < 1e-7:
                continue
            robot = next((a for a in actors if a.startswith("/World/G1/")), None)
            cup = any("/cup_V1" in a for a in actors)
            environment = next((a for a in actors if not a.startswith("/World/G1/")), None)
            floor_contact = (
                robot
                and "ankle_roll_link" in robot
                and environment
                and (
                    environment in self.floor_actor_paths
                    or any(word in environment.rsplit("/", 1)[-1].lower() for word in ("floor", "ground"))
                )
                and all(abs(float(c.position[2])) < 0.08 for c in points)
            )
            kind = (
                "cup_hand"
                if cup and robot and "_hand_" in robot
                else "foot_floor_support"
                if floor_contact
                else "robot_environment"
                if robot and environment
                else "cup_environment"
                if cup
                else "other"
            )
            if key not in self.pairs:
                self.pairs[key] = dict(
                    actors=actors,
                    colliders=colliders,
                    kind=kind,
                    first_time=time,
                    first_phase=phase,
                    reports=0,
                    max_impulse_Ns=0.0,
                    maximum_episode_impulse_Ns=0.0,
                    maximum_unexpected_episode_impulse_Ns=0.0,
                    minimum_separation_m=float("inf"),
                    phases={},
                )
            item = self.pairs[key]
            same_step = item.get("last_time") == time
            step_impulse = (item.get("last_step_impulse_Ns", 0) if same_step else 0) + impulse
            vertical_force = (item.get("last_vertical_force_N", 0) if same_step else 0) + abs(
                sum(float(tuple(c.impulse)[2]) for c in points)
            ) / self.dt
            phase_item = item["phases"].setdefault(
                phase, dict(first_time=time, last_time=time, max_impulse_Ns=0.0, reports=0)
            )
            phase_item.update(
                last_time=time,
                max_impulse_Ns=max(phase_item["max_impulse_Ns"], step_impulse),
                reports=phase_item["reports"] + 1,
            )
            item.update(
                last_time=time,
                last_phase=phase,
                reports=item["reports"] + 1,
                last_vertical_force_N=vertical_force,
                last_step_impulse_Ns=step_impulse,
                max_impulse_Ns=max(item["max_impulse_Ns"], step_impulse),
                minimum_separation_m=min(
                    item["minimum_separation_m"], *(float(c.separation) for c in points)
                ),
            )
            if phase != "setup":
                item["maximum_episode_impulse_Ns"] = max(item["maximum_episode_impulse_Ns"], step_impulse)
                if kind == "robot_environment":
                    item["maximum_unexpected_episode_impulse_Ns"] = max(
                        item["maximum_unexpected_episode_impulse_Ns"], step_impulse
                    )

    def recent_cup_contact(self, actor, time, maximum_age=0.10):
        return any(
            x["kind"] == "cup_environment" and actor in x["actors"] and time - x["last_time"] <= maximum_age
            for x in self.pairs.values()
        )

    def active_cup_contact(self, actor):
        return any(
            state["active"]
            and actor in state["actors"]
            and any("/cup_V1" in path for path in state["actors"])
            and key in self.pairs
            and self.pairs[key]["kind"] == "cup_environment"
            for key, state in self.contact_states.items()
        )

    def recent_cup_support_force(self, actor, time, maximum_age=0.02):
        return sum(
            x.get("last_vertical_force_N", 0)
            for x in self.pairs.values()
            if x["kind"] == "cup_environment"
            and actor in x["actors"]
            and 0 <= time - x["last_time"] <= maximum_age
        )

    def export(self):
        return dict(
            physics_dt=self.dt,
            pairs=list(self.pairs.values()),
            callback_errors=list(self.errors),
            contact_states=list(self.contact_states.values()),
            unexpected_robot_environment_pairs=sum(
                x["maximum_unexpected_episode_impulse_Ns"] > 0.0005 for x in self.pairs.values()
            ),
            note="Impulses are summed contact-point magnitudes per physics step; this is a contact ledger, not a global collision-free certificate.",
        )
