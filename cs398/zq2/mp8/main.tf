variable "project" {
  type = "string"
  // Change the following line to the correct GCP project!
  default = "cs398-zq2-mp8"
}

provider "google" {
  credentials = "${file("account.json")}"
  project     = "${var.project}"
  region      = "us-central1"
}

// Problem 2 - Creating a storage bucket
resource "google_storage_bucket" "file-storage" {
  name        = "cs398-file-storage-zq2"
  location    = "US"

  versioning {
    enabled   = true
  }
}

// Problem 3 - Creating an Instance
resource "google_compute_instance" "nebula-in-the-cloud" {
  name        = "nebula-in-the-cloud"
  machine_type= "n1-standard-1"
  zone        = "us-central1-a"

  description = "Look, we're cooking with clouds now!"

  tags = ["cs398", "mp8"]

  boot_disk {
      initialize_params {
        image = "ubuntu-os-cloud/ubuntu-1604-lts"
      }
    }

  network_interface {
    network = "default"
      access_config {
        nat_ip = "${google_compute_address.nebula-in-the-cloud-address.address}"
        }
    }

  attached_disk {
      source = "${google_compute_disk.dataset-disk.self_link}"
  }

  metadata_startup_script = "${file("startup.sh")}"
}

// Problem 4 - Creating a Disk and Attaching It
resource "google_compute_disk" "dataset-disk" {
  name        = "dataset-disk"
  zone        = "us-central1-a"
  size        = 10
}

// Problem 5 - Making Your Instance Accessible

resource "google_compute_address" "nebula-in-the-cloud-address" {
  name = "nebula-in-the-cloud-address"
}

resource "google_compute_firewall" "nebula-in-the-cloud-firewall" {
  name    = "nebula-in-the-cloud-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

}
