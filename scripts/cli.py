#!/usr/bin/env python3
# coding: utf-8

"""up-model 统一命令行入口。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="up-model",
        description="城市规划：OSM/合成拓扑、可视化、Gurobi 静态规划与 RL 轨迹数据",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    from scripts import topology_osm
    from scripts import batch_solve
    from scripts import mip_land_use
    from scripts import mip_poi_placement
    from scripts import mip_road_planning
    from scripts import prepare_training_data
    from scripts import topology_synthetic
    from scripts import visualize_training_data

    topology_osm.add_subcommand(sub)
    prepare_training_data.add_subcommand(sub)

    topology_synthetic.add_subcommand(sub)
    try:
        from scripts import sumo_collect
        from scripts import sumo_demand
        from scripts import sumo_replay

        sumo_demand.add_subcommand(sub)
        sumo_collect.add_subcommand(sub)
        sumo_replay.add_subcommand(sub)
    except ImportError:
        pass

    visualize_training_data.add_subcommand(sub)
    visualize_training_data.add_selection_server_subcommand(sub)

    mip_land_use.add_subcommand(sub)
    mip_road_planning.add_subcommand(sub)
    mip_poi_placement.add_subcommand(sub)
    
    batch_solve.add_subcommand(sub)
   

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
