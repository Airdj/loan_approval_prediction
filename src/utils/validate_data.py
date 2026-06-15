import great_expectations as ge

def validate_loan_data(df, columns):
    print('Starting data validation with Great Expectations')
    print(ge.__version__)
    context = ge.get_context()

    data_source_name = 'my_data_source'
    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset_name = 'my_data_asset_name'
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition_name = 'my_batch_definition'
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name=batch_definition_name)
    batch_parameters = {'dataframe': df}

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = context.suites.add(ge.ExpectationSuite(name='loan_expectation_suite'))
    #schema validation - essential columns
    for col in columns:
        exceptation1 = ge.expectations.ExpectColumnToExist(column=col)
        suite.add_expectation(exceptation1)
    # bussines logic validation
    my_expectation_list = [

        ge.expectations.ExpectColumnValuesToBeBetween(
            column="person_age",
            min_value=18,
            max_value=99
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="person_income",
            min_value=0,
            strict_min=True
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='person_home_ownership',
            value_set=['RENT','MORTGAGE','OWN','OTHER']
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="person_emp_length",
            min_value=1,
            max_value=60
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='loan_intent',
            value_set=['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT']
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='loan_grade',
            value_set=['A','B','C','D','E','F','G']
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="loan_amnt",
            min_value=1000,
            max_value=10000000
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="loan_int_rate",
            min_value=0.1,
            max_value=50.0
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="loan_percent_income",
            min_value=0.0,
            max_value=1.0
        ),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='cb_person_default_on_file',
            value_set=['N', 'Y']
        ),
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="cb_person_cred_hist_length",
            min_value=0,
            max_value=99
        ),]

    for expectation in my_expectation_list:
        suite.add_expectation(expectation)

    # run validation
    print('running validation')
    validation_definition = ge.ValidationDefinition(data=batch_definition, suite=suite, name='validation_batch_definition')
    validation_definition = context.validation_definitions.add(validation_definition)
    results = validation_definition.run(batch_parameters=batch_parameters)

    #process results
    failed_expectations = []
    failed_columns = []
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['type']
            expectation_column = r['expectation_config']['kwargs']['column']
            failed_expectations.append(expectation_type)
            failed_columns.append(expectation_column)
    #print validation summary
    total_checks = len(results['results'])
    passed_checks = sum(1 for r in results['results'] if r['success'])
    failed_checks = total_checks - passed_checks

    if results['success']:
        print(f'Data validation PASSED: {passed_checks}/{total_checks} checks successful')
    else:
        print(f'Data validation FAILED: {failed_checks}/{total_checks} checks failed')
        print(f'Failed expectations: {failed_expectations}')
        print(f'Failed columns: {failed_columns}')
    return results['success'], failed_expectations