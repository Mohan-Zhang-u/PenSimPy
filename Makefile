GEMFURY_AUTH_TOKEN := ${GEMFURY_AUTH_TOKEN}
PROJECT_NAME := pensimpy
PROJECT_HOME_DIR := $(shell pwd)/${PROJECT_NAME}

# distribution details
VERSION := $(shell awk '$$1 == "__version__" {print $$NF}'  ${PROJECT_HOME_DIR}/_version.py)
OS := none
CPU_ARCH = any

help:
	@echo "PenSimPy Makefile Help:\n"\
	"clean:  Remove all cache and wheel packages.\n"\
	"build:  Build POV-core wheel package via setup.py.\n"\
	"version:  Show current POV-core version.\n"\
	"publish:  Upload the package in dist directory that matches current POV-core version.\n"\
	" VERSION Specify another version to upload (If there is one available). e.g. make publish VERSION=1.0.1\n"\
	"tests:  Run all test cases under POV-core project.\n"\
	" TESTS: The path to the module where contains the test cases you want to run. e.g make tests TESTS=./pov_core/a_module"

clean-dist:
	rm -r ./dist 2>/dev/null || true

clean-cache:
	rm -r *.egg-info || true
	python3 setup.py clean --all

clean: clean-dist clean-cache

build: clean
	python3 setup.py bdist_wheel

version:
	@echo $(VERSION)

publish: override VERSION := $(if $(VERSION),$(VERSION),)
publish: WHEEL_FILENAME := $(PROJECT_NAME)-$(VERSION)-py3-$(OS)-$(CPU_ARCH).whl
publish:
	curl -F package=@dist/$(WHEEL_FILENAME) https://$(GEMFURY_AUTH_TOKEN)@push.fury.io/quartic-ai/

TESTS := $(PROJECT_HOME_DIR)

tests: override TESTS := $(if $(TESTS),$(TESTS),)
tests:
	nosetests ${TESTS}