// This pipeline triggers multiple Milvus data verify tests with different configurations
// It covers various scenarios including different schemas, concurrent operations, and storage versions

pipeline {
    options {
        timestamps()
        timeout(time: 2000, unit: 'MINUTES')   // Extended timeout for batch tests
    }
    agent {
        kubernetes {
            inheritFrom 'default'
            defaultContainer 'main'
            yamlFile "jenkins/pods/validation-test-client.yaml"
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
        booleanParam(
            description: 'Test E-commerce Schema',
            name: 'test_ecommerce',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Document Schema',
            name: 'test_document',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Multimedia Schema',
            name: 'test_multimedia',
            defaultValue: true
        )
        booleanParam(
            description: 'Test IoT Schema',
            name: 'test_iot',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Social Schema',
            name: 'test_social',
            defaultValue: false
        )
        booleanParam(
            description: 'Test All Data Types Schema',
            name: 'test_all_datatypes',
            defaultValue: false
        )
        booleanParam(
            description: 'Test Storage V1',
            name: 'test_storage_v1',
            defaultValue: true
        )
        booleanParam(
            description: 'Test Storage V2',
            name: 'test_storage_v2',
            defaultValue: true
        )
        string(
            description: 'Test Duration in Seconds per Schema',
            name: 'duration',
            defaultValue: '900'
        )
        booleanParam(
            description: 'Include Vector Fields in PostgreSQL Operations',
            name: 'include_vector',
            defaultValue: true
        )
        booleanParam(
            description: 'Keep Environment After Tests',
            name: 'keep_env',
            defaultValue: false
        )
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        NAMESPACE = "chaos-testing"
    }
    
    stages {
        stage('Prepare Test Scenarios') {
            steps {
                script {
                    // Define all test scenarios
                    def allScenarios = []
                    
                    // Schema presets to test based on parameters
                    def schemaPresets = []
                    if (params.test_ecommerce) schemaPresets.add('ecommerce')
                    if (params.test_document) schemaPresets.add('document') 
                    if (params.test_multimedia) schemaPresets.add('multimedia')
                    if (params.test_iot) schemaPresets.add('iot')
                    if (params.test_social) schemaPresets.add('social')
                    if (params.test_all_datatypes) schemaPresets.add('all_datatypes')
                    
                    // Test configurations with different thread counts
                    def testConfigs = [
                        [threads: 2, compareInterval: 30, desc: 'Light Load'],
                        [threads: 4, compareInterval: 60, desc: 'Normal Load'],
                        [threads: 8, compareInterval: 90, desc: 'Heavy Load']
                    ]
                    
                    // Storage versions
                    def storageVersions = []
                    if (params.test_storage_v1) {
                        storageVersions.add('V1')
                    }
                    if (params.test_storage_v2) {
                        storageVersions.add('V2')
                    }
                    
                    // Build test matrix
                    schemaPresets.each { schema ->
                        testConfigs.each { testConfig ->
                            storageVersions.each { storage ->
                                allScenarios.add([
                                    schemaPreset: schema,
                                    threads: testConfig.threads,
                                    compareInterval: testConfig.compareInterval,
                                    testDesc: testConfig.desc,
                                    storage: storage,
                                    duration: params.duration
                                ])
                            }
                        }
                    }
                    
                    env.TEST_SCENARIOS = groovy.json.JsonOutput.toJson(allScenarios)
                    echo "Total test scenarios: ${allScenarios.size()}"
                    echo "Test scenarios: ${env.TEST_SCENARIOS}"
                }
            }
        }
        
        stage('Execute Test Scenarios') {
            steps {
                script {
                    def scenarios = groovy.json.JsonSlurper().parseText(env.TEST_SCENARIOS)
                    def parallelTests = [:]
                    
                    scenarios.eachWithIndex { scenario, index ->
                        def testName = "Test-${index + 1}: ${scenario.schemaPreset}-${scenario.testDesc}-${scenario.storage}"
                        
                        parallelTests[testName] = {
                            stage(testName) {
                                echo "Starting test: ${testName}"
                                echo "Configuration: ${groovy.json.JsonOutput.toJson(scenario)}"
                                
                                try {
                                    build job: 'pymilvus_pg_stable_test', 
                                        parameters: [
                                            string(name: 'image_repository', value: params.image_repository),
                                            string(name: 'image_tag', value: params.image_tag),
                                            string(name: 'schema_preset', value: scenario.schemaPreset),
                                            string(name: 'duration', value: scenario.duration.toString()),
                                            string(name: 'threads', value: scenario.threads.toString()),
                                            string(name: 'compare_interval', value: scenario.compareInterval.toString()),
                                            string(name: 'storage_version', value: scenario.storage),
                                            booleanParam(name: 'include_vector', value: params.include_vector),
                                            booleanParam(name: 'keep_env', value: params.keep_env)
                                        ],
                                        wait: true,
                                        propagate: false
                                    
                                    echo "Test ${testName} completed successfully"
                                } catch (Exception e) {
                                    echo "Test ${testName} failed: ${e.getMessage()}"
                                    currentBuild.result = 'UNSTABLE'
                                }
                            }
                        }
                    }
                    
                    // Execute tests in batches to avoid overwhelming the system
                    def batchSize = 4  // Run 4 tests in parallel
                    def batches = []
                    parallelTests.each { name, test ->
                        if (batches.size() == 0 || batches[-1].size() >= batchSize) {
                            batches.add([:])
                        }
                        batches[-1][name] = test
                    }
                    
                    batches.eachWithIndex { batch, batchIndex ->
                        stage("Batch ${batchIndex + 1}") {
                            parallel batch
                        }
                    }
                }
            }
        }
        
        stage('Collect Results') {
            steps {
                script {
                    echo "All test scenarios completed"
                    echo "Collecting test results and metrics..."
                    
                    // Archive test results
                    sh """
                    mkdir -p ${env.ARTIFACTS}
                    echo "Test Summary:" > ${env.ARTIFACTS}/test_summary.txt
                    echo "Total Scenarios: ${groovy.json.JsonSlurper().parseText(env.TEST_SCENARIOS).size()}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "Parameters:" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - E-commerce Schema: ${params.test_ecommerce}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Document Schema: ${params.test_document}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Multimedia Schema: ${params.test_multimedia}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - IoT Schema: ${params.test_iot}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Social Schema: ${params.test_social}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - All Data Types Schema: ${params.test_all_datatypes}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Duration per Schema: ${params.duration}s" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Include Vector: ${params.include_vector}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V1: ${params.test_storage_v1}" >> ${env.ARTIFACTS}/test_summary.txt
                    echo "  - Storage V2: ${params.test_storage_v2}" >> ${env.ARTIFACTS}/test_summary.txt
                    """
                    
                    archiveArtifacts artifacts: "_artifacts/**", allowEmptyArchive: true
                }
            }
        }
    }
    
    post {
        always {
            echo 'Batch data verify pipeline completed'
        }
        success {
            echo 'All data verify scenarios executed successfully!'
        }
        unstable {
            echo 'Some data verify scenarios failed. Check individual test results.'
        }
        failure {
            echo 'Batch data verify pipeline failed'
        }
    }
}