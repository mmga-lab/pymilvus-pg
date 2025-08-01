// Enhanced pipeline for Milvus data verify with comprehensive configurations
// Supports multiple schema presets, storage versions, and data consistency verification

pipeline {
    options {
        timestamps()
        timeout(time: 1000, unit: 'MINUTES')
    }
    agent {
        kubernetes {
            cloud '4am'
            defaultContainer 'main'
            yamlFile 'jenkins/pods/validation-test-client.yaml'
            customWorkspace '/home/jenkins/agent/workspace'
            idleMinutes 5
        }
    }
    parameters {
        string(
            description: 'Image Repository',
            name: 'image_repository',
            defaultValue: 'harbor.milvus.io/milvus/milvus'
        )
        string(
            description: 'Image Tag',
            name: 'image_tag',
            defaultValue: 'master-latest'
        )
        string(
            description: 'QueryNode Nums',
            name: 'querynode_nums',
            defaultValue: '3'
        )
        string(
            description: 'DataNode Nums',
            name: 'datanode_nums',
            defaultValue: '3'
        )
        string(
            description: 'Proxy Nums',
            name: 'proxy_nums',
            defaultValue: '1'
        )
        booleanParam(
            description: 'Keep Environment',
            name: 'keep_env',
            defaultValue: false
        )
        choice(
            description: '''Built-in schema preset for Milvus data verify. Available schemas:

BUSINESS DOMAIN SCHEMAS:
• ecommerce - E-commerce product catalog with multi-modal embeddings (10 fields, dynamic: True)
• document - Document storage for RAG applications (9 fields, dynamic: True)
• multimedia - Multimedia content storage with multi-modal embeddings (8 fields, dynamic: True)
• social - Social media user profiles with embeddings (9 fields, dynamic: True)

SPECIALIZED SCHEMAS:
• iot - IoT sensor data with time-series support (8 fields, dynamic: True)
• all_datatypes - Schema demonstrating all Milvus data types with nullable and default examples (10 fields, dynamic: False)

Each schema includes vector fields with appropriate dimensions and various data types for comprehensive data verification.''',
            name: 'schema_preset',
            choices: [
                'ecommerce',        // E-commerce product catalog with multi-modal embeddings
                'document',         // Document storage for RAG applications
                'multimedia',       // Multimedia content storage with multi-modal embeddings
                'iot',              // IoT sensor data with time-series support
                'social',           // Social media user profiles with embeddings
                'all_datatypes'     // Schema demonstrating all Milvus data types
            ]
        )
        string(
            description: 'Test Duration in Seconds (0 means run indefinitely)',
            name: 'duration',
            defaultValue: '1800'
        )
        string(
            description: 'Number of Writer Threads',
            name: 'threads',
            defaultValue: '4'
        )
        string(
            description: 'Seconds Between Validation Checks',
            name: 'compare_interval',
            defaultValue: '60'
        )
        booleanParam(
            description: 'Include Vector Fields in PostgreSQL Operations',
            name: 'include_vector',
            defaultValue: false
        )
        booleanParam(
            description: 'Use Existing Instances (skip deployment of new Milvus and PostgreSQL)',
            name: 'use_existing_instances',
            defaultValue: false
        )
        string(
            description: 'Existing Milvus URI (only used when use_existing_instances is true)',
            name: 'existing_milvus_uri',
            defaultValue: 'http://10.104.17.43:19530'
        )
        string(
            description: 'Existing PostgreSQL Connection String (only used when use_existing_instances is true)',
            name: 'existing_pg_conn',
            defaultValue: 'postgresql://postgres:admin@10.104.20.96:5432/postgres'
        )
        choice(
            description: 'Storage Version',
            name: 'storage_version',
            choices: ['V2', 'V1']
        )
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        RELEASE_NAME = "data-verify-${env.BUILD_ID}"
        NAMESPACE = "chaos-testing"
        POSTGRES_HOST = "postgres-service"
        POSTGRES_DB = "milvus_data_verify"
        POSTGRES_USER = "postgres"
        POSTGRES_PASSWORD = "postgres"
    }

    stages {
        stage('Install Dependencies') {
            steps {
                container('main') {
                    script {
                        sh "pip install pdm"
                        sh "pip install uv"
                    }
                }
            }
        }
        stage('Install milvus-ingest') {
            steps {
                container('main') {
                    script {
                        sh """
                        # Set UV cache directory to use NFS mounted path
                        export UV_CACHE_DIR=/tmp/.uv-cache
                        
                        # Install PDM if not available
                        which pdm || pip install pdm
                        
                        # Use Python 3.10 specifically
                        pdm use python3.10
                        pdm config use_uv true
                        
                        # Install milvus-ingest from current workspace
                        # Fix lockfile if needed
                        # pdm lock --update-reuse || true
                        rm -rf pdm.lock
                        pdm install
                        
                        # Verify installation
                        pdm run pymilvus-pg --help
                        """
                    }
                }
            }
        }        

        stage('Prepare Milvus Values') {
            when {
                not { params.use_existing_instances }
            }
            steps {
                container('main') {
                    script {
                        sh """
                        # Create working directory for values
                        mkdir -p /tmp/milvus-values
                        
                        # Select appropriate values file based on storage version (cluster mode only)
                        if [ "${params.storage_version}" = "V2" ]; then
                            cp jenkins/values/cluster-storagev2.yaml /tmp/milvus-values/values.yaml
                            echo "Using cluster Storage V2 configuration"
                        else
                            cp jenkins/values/cluster-storagev1.yaml /tmp/milvus-values/values.yaml
                            echo "Using cluster Storage V1 configuration"
                        fi
                        
                        # Customize values based on parameters
                        cd /tmp/milvus-values
                        
                        # Update node replicas
                        yq -i '.queryNode.replicas = "${params.querynode_nums}"' values.yaml
                        yq -i '.dataNode.replicas = "${params.datanode_nums}"' values.yaml
                        yq -i '.proxy.replicas = "${params.proxy_nums}"' values.yaml
                        
                        echo "Final values configuration:"
                        cat values.yaml
                        """
                    }
                }
            }
        }

        stage('Deploy Milvus') {
            when {
                not { params.use_existing_instances }
            }
            options {
                timeout(time: 15, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        def image_tag_modified = ''
                        
                        if ("${params.image_tag}" =~ 'latest') {
                            image_tag_modified = sh(returnStdout: true, script: "tagfinder get-tag -t ${params.image_tag}").trim()
                        }
                        else {
                            image_tag_modified = "${params.image_tag}"
                        }
                        
                        sh 'helm repo add milvus https://zilliztech.github.io/milvus-helm'
                        sh 'helm repo update'
                        
                        sh """
                        cd /tmp/milvus-values
                        
                        echo "Deploying Milvus cluster with configuration:"
                        echo "Image Repository: ${params.image_repository}"
                        echo "Image Tag: ${params.image_tag}"
                        echo "Resolved Image Tag: ${image_tag_modified}"
                        echo "Storage Version: ${params.storage_version}"
                        
                        helm install --wait --debug --timeout 600s ${env.RELEASE_NAME} milvus/milvus \\
                            --set image.all.repository=${params.image_repository} \\
                            --set image.all.tag=${image_tag_modified} \\
                            --set metrics.serviceMonitor.enabled=true \\
                            --set quotaAndLimits.enabled=false \\
                            -f values.yaml -n=${env.NAMESPACE}
                        """
                        sh 'cat /tmp/milvus-values/values.yaml'
                        sh "kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl wait --for=condition=Ready pod -l release=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME}"
                    }   
                }
            }
        }

        stage('Setup PostgreSQL') {
            when {
                not { params.use_existing_instances }
            }
            options {
                timeout(time: 15, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        sh """
                        echo "Setting up PostgreSQL for validation testing..."
                        
                        # Deploy PostgreSQL if not exists
                        helm repo add bitnami https://charts.bitnami.com/bitnami || true
                        helm repo update
                        
                        # Check if PostgreSQL is already deployed
                        if ! helm list -n ${env.NAMESPACE} | grep -q postgres-${env.BUILD_ID}; then
                            echo "Deploying PostgreSQL instance..."
                            helm install postgres-${env.BUILD_ID} bitnami/postgresql \\
                                --set auth.postgresPassword=${env.POSTGRES_PASSWORD} \\
                                --set auth.database=${env.POSTGRES_DB} \\
                                --set primary.persistence.enabled=false \\
                                --set volumePermissions.enabled=true \\
                                --wait --timeout=600s \\
                                -n ${env.NAMESPACE}
                        else
                            echo "PostgreSQL already deployed"
                        fi
                        
                        # Wait for PostgreSQL to be ready
                        kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=postgresql,app.kubernetes.io/instance=postgres-${env.BUILD_ID} -n ${env.NAMESPACE} --timeout=300s
                        
                        echo "PostgreSQL setup completed"
                        """
                    }
                }
            }
        }
        
        stage('Run PyMilvus-PG Validation') {
            options {
                timeout(time: 120, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        // Determine connection strings based on whether using existing instances
                        def milvusUri = ""
                        def pgConn = ""
                        
                        if ("${params.use_existing_instances}" == "true") {
                            milvusUri = "${params.existing_milvus_uri}"
                            pgConn = "${params.existing_pg_conn}"
                            echo "Using existing instances:"
                            echo "Existing Milvus URI: ${milvusUri}"
                            echo "Existing PostgreSQL Connection: ${pgConn}"
                        } else {
                            def milvusHost = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            def postgresHost = sh(returnStdout: true, script: "kubectl get svc/postgres-${env.BUILD_ID}-postgresql -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${milvusHost}:19530"
                            pgConn = "postgresql://${env.POSTGRES_USER}:${env.POSTGRES_PASSWORD}@${postgresHost}:5432/${env.POSTGRES_DB}"
                            echo "Using deployed instances:"
                            echo "Milvus Host: ${milvusHost}"
                            echo "PostgreSQL Host: ${postgresHost}"
                        }
                        
                        sh """
                        echo "Starting Milvus data verify with configuration:"
                        echo "Schema Preset: ${params.schema_preset}"
                        echo "Duration: ${params.duration} seconds"
                        echo "Threads: ${params.threads}"
                        echo "Compare Interval: ${params.compare_interval} seconds"
                        echo "Include Vector: ${params.include_vector}"
                        echo "Storage Version: ${params.storage_version}"
                        echo "Using Existing Instances: ${params.use_existing_instances}"
                        echo "Milvus URI: ${milvusUri}"
                        echo "PostgreSQL Connection: ${pgConn}"
                        
                        # Set environment variables for PyMilvus-PG
                        export MILVUS_URI="${milvusUri}"
                        export PG_CONN="${pgConn}"
                        
                        echo "Environment variables set:"
                        echo "MILVUS_URI: \$MILVUS_URI"
                        echo "PG_CONN: \$PG_CONN"
                        
                        # Show available schemas
                        echo "Available schema presets:"
                        pdm run pymilvus-pg list-schemas
                        
                        echo "Using schema preset: ${params.schema_preset}"
                        pdm run pymilvus-pg show-schema ${params.schema_preset}
                        
                        # Run PyMilvus-PG ingest command with specified parameters
                        echo "Starting PyMilvus-PG ingest validation..."
                        COLLECTION_NAME="verify_${params.schema_preset}_${env.BUILD_ID}"
                        
                        pdm run pymilvus-pg ingest \\
                            --threads ${params.threads} \\
                            --compare-interval ${params.compare_interval} \\
                            --duration ${params.duration} \\
                            --collection "\$COLLECTION_NAME" \\
                            --drop-existing \\
                            --schema ${params.schema_preset} \\
                            \$([[ "${params.include_vector}" == "true" ]] && echo "--include-vector" || echo "")
                        
                        echo "Milvus data verify completed successfully"
                        """
                    }
                }
            }
        }
        
        stage('Final Validation Check') {
            options {
                timeout(time: 30, unit: 'MINUTES')
            }
            steps {
                container('main') {
                    script {
                        // Determine connection strings based on whether using existing instances
                        def milvusUri = ""
                        def pgConn = ""
                        
                        if ("${params.use_existing_instances}" == "true") {
                            milvusUri = "${params.existing_milvus_uri}"
                            pgConn = "${params.existing_pg_conn}"
                            echo "Using existing instances for final validation:"
                            echo "Existing Milvus URI: ${milvusUri}"
                            echo "Existing PostgreSQL Connection: ${pgConn}"
                        } else {
                            def milvusHost = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            def postgresHost = sh(returnStdout: true, script: "kubectl get svc/postgres-${env.BUILD_ID}-postgresql -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            milvusUri = "http://${milvusHost}:19530"
                            pgConn = "postgresql://${env.POSTGRES_USER}:${env.POSTGRES_PASSWORD}@${postgresHost}:5432/${env.POSTGRES_DB}"
                            echo "Using deployed instances for final validation:"
                            echo "Milvus Host: ${milvusHost}"
                            echo "PostgreSQL Host: ${postgresHost}"
                        }
                        
                        sh """
                        echo "Running final validation check:"
                        echo "Schema Preset: ${params.schema_preset}"
                        echo "Using Existing Instances: ${params.use_existing_instances}"
                        echo "Milvus URI: ${milvusUri}"
                        echo "PostgreSQL Connection: ${pgConn}"
                        
                        # Set environment variables for PyMilvus-PG
                        export MILVUS_URI="${milvusUri}"
                        export PG_CONN="${pgConn}"
                        
                        # Determine collection name based on schema preset
                        COLLECTION_NAME="verify_${params.schema_preset}_${env.BUILD_ID}"
                        
                        echo "Running final validation on collection: \$COLLECTION_NAME"
                        
                        # Run final validation using validate command
                        pdm run pymilvus-pg validate \\
                            --collection "\$COLLECTION_NAME" \\
                            --sample-percentage 20.0 \\
                            \$([[ "${params.include_vector}" == "true" ]] && echo "--include-vector" || echo "")
                        
                        echo "Final validation completed successfully"
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Upload logs and cleanup'
            container('main') {
                script {
                    echo "Get pod status"
                    sh "kubectl get pods -o wide|grep ${env.RELEASE_NAME} || true"
                    
                    // Collect logs using kubectl
                    sh """
                    mkdir -p k8s_log/${env.RELEASE_NAME}
                    kubectl logs -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --all-containers=true --tail=-1 > k8s_log/${env.RELEASE_NAME}/milvus-logs.txt || true
                    kubectl describe pods -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} > k8s_log/${env.RELEASE_NAME}/pod-descriptions.txt || true
                    """

                    // Archive logs
                    sh "tar -zcvf artifacts-${env.RELEASE_NAME}-server-logs.tar.gz k8s_log/ --remove-files || true"

                    archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-server-logs.tar.gz", allowEmptyArchive: true


                    if ("${params.keep_env}" == "false" && "${params.use_existing_instances}" == "false") {
                        sh "helm uninstall ${env.RELEASE_NAME} -n ${env.NAMESPACE} || true"
                        sh "helm uninstall postgres-${env.BUILD_ID} -n ${env.NAMESPACE} || true"
                    } else if ("${params.use_existing_instances}" == "true") {
                        echo "Skipping cleanup - using existing instances"
                    }
                }
            }
        }
        success {
            echo 'Milvus data verify completed successfully!'
            container('main') {
                script {
                    echo "Data Verify Summary:"
                    echo "Schema Preset: ${params.schema_preset}"
                    echo "Duration: ${params.duration} seconds"
                    echo "Threads: ${params.threads}"
                    echo "Compare Interval: ${params.compare_interval} seconds"
                    echo "Include Vector: ${params.include_vector}"
                    echo "Storage Version: ${params.storage_version}"

                    if ("${params.keep_env}" == "false" && "${params.use_existing_instances}" == "false") {
                        sh "helm uninstall ${env.RELEASE_NAME} -n ${env.NAMESPACE} || true"
                        sh "helm uninstall postgres-${env.BUILD_ID} -n ${env.NAMESPACE} || true"
                    } else if ("${params.use_existing_instances}" == "true") {
                        echo "Skipping cleanup - using existing instances"
                    }
                }
            }
        }
        unstable {
            echo 'Data verify completed with some issues'
        }
        failure {
            echo 'Data verify failed'
        }
        changed {
            echo 'Data verify results changed from previous run'
        }
    }
}
