---
marp: true
theme: cscs
# class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('../slides-support/common/4k-slide-bg-white.png')
size: 16:9
---

# **Userlab Day 2024**
![bg cover](../slides-support/common/title-bg3.png)
<!-- _paginate: skip  -->
<!-- _class: titlecover -->
<!-- _footer: "" -->

### NCCL

#### CSCS

---

# NVIDIA Collective Communication Library (NCCL)

- inter-GPU (collective) communication primitives

|  Send/Recv   | Broadcast     | Reduce | AllReduce | AllGather | ReduceScatter |
|---|---|---|---|---|---|

- single kernel implementation for communication and computation
- topology-aware, supports NVLINK
- API is similar to MPI
- bootstrap for parallel environment is out-of-band (not provided)
- can be used in combination with MPI

---

# Communicators and Communication Groups

- communication primitives transfer data among members of a communication `group`
- each group member corresponds to a CUDA device index
- a `communicator` refers to a particular `group`


---
