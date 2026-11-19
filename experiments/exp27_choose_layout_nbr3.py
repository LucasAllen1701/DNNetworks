"""Experiment 27: choosing the image layout with neighbour pairs plus upstart among the choices.

Experiment 22 chose between upstart, cover and nbr3_cover on the image tasks,
but experiment 21's best layout overall, nbr3_upstart (neighbouring-pixel
pair ANDs in layer 1, upstart in layer 2, h = 3), was not among the choices.
Same method as experiment 22 (held-out training streams, or fewest helpers,
compared with each fixed layout and the oracle), image tasks only, with four
layouts: upstart, cover, nbr3_cover, nbr3_upstart. The rows of the first
three should equal experiment 22's image rows, and nbr3_upstart's test
columns experiment 21's.
Run from the repository root:  python experiments/exp27_choose_layout_nbr3.py
Writes experiments/results/exp27_choose_layout_nbr3.csv.
"""
import exp22_choose_layout as e22
import exp19_small_images as e19

e22.FAMILIES = {"images": dict(tasks=e19.TASKS, sizes=e19.SIZES, layouts=["upstart", "cover", "nbr3_cover", "nbr3_upstart"],
                               n_in=e19.N_IN)}


def main():
    e22.main(families=("images",), name="exp27_choose_layout_nbr3.csv")   # e22.main's default was fixed at import


if __name__ == "__main__":
    main()
