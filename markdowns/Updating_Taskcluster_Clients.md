# Taskcluster Client Dependencies
This notebook give an overview of the various repositories using `environment variables` and `configuration files`.

We shall be having a look at the following repositories and understanding how do they get the credentials.
- [taskcluster-cli](https://github.com/taskcluster/taskcluster-cli)
- [taskcluster-client.py](https://github.com/taskcluster/taskcluster-client.py)
- [taskcluster-client](https://github.com/taskcluster/taskcluster-client)
- [taskcluster-client-go](https://github.com/taskcluster/taskcluster-client-go)
- [taskcluster-client-java](https://github.com/taskcluster/taskcluster-client-java)

|                       | client-java | cli | client-go | client.py | client |
|:---------------------:|:-----------:|:---:|:---------:|:---------:|:------:|
|     Configuration Files     |             |  ✔️  |           |          |        |
| Environment Variables |      ✔️      |  ✔️   |     ✔️     |      ✔️     |    ✔️   |

## Making infographics
We shall now be looking at various insights which can be generated from these *repositories*.

